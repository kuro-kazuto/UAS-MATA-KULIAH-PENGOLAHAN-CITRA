# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 17:35:09 2021

@author: galih-ap
"""

import sys
import PySimpleGUI as sg
import gzip, os, sys
import numpy as np
from scipy.stats import multivariate_normal
from urllib.request import urlretrieve
import matplotlib.pyplot as plt
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from skimage.transform import ProjectiveTransform
from skimage.io import imread
from PIL import Image, ImageTk

def start() :
    if not sys.platform.startswith('win'):
        sg.popup_error('Sorry, you gotta be on Windows')
        sys.exit()
    import winsound
    
    sg.theme('DarkGreen3')

    layout = [[sg.Text('GUI Untuk Program Autoencoder Variasi Untuk Merekonstruksi dan Menghasilkan Gambar ')],
              [sg.Text('Nama : GALIH AJI PAMBUDI ')],
              [sg.Text('NIM    : 3332180058 ')],
              [sg.Text('github : github.com/kuro-kazuto ')],
              [sg.Text('')],
              [sg.Button('Mulai Program', button_color=('white', 'green'), key='start'),
               sg.Button('Gambar Asal', button_color=('white', 'black'), key='GA'),
               sg.Button('Rekonstruksi VAE', button_color=('white', 'black'),size=(20, 1), key='GRV'),
               sg.Button('Generated VAE', button_color=('white', 'black'),size=(20, 1), key='GGV'),
               sg.Button('Grafik Sebaran', button_color=('white', 'red'), key='grafik')],
              [sg.Button('Generated 2D VAE', button_color=('white', 'green'),size=(22, 1), key='G2V'),
               sg.Button('Latent Space 2D VAE', button_color=('black', 'yellow'),size=(36, 1), key='L2V'),
               sg.Button('Grafik Sebaran 2D VAE', button_color=('white', 'red'),size=(22, 1), key='G2A')]
              ]

    window = sg.Window("UAS PENGOLAHAN CITRA 2021", layout, auto_size_buttons=False, default_button_element_size=(12,1), use_default_focus=False, finalize=True)


    recording = have_data = False
    while True:
        event, values = window.read(timeout=100)
        if event == sg.WINDOW_CLOSED:
            break
        
        elif event == 'start':
            winsound.PlaySound("ButtonClick.wav", 1) if event != sg.TIMEOUT_KEY else None
            sg.popup('Program Akan Mencari Dataset dan Merekontruksinya','Mungkin GUI akan Lag/Not Responding','Harap Tunggu Beberapa Menit Hingga Selesai','','Klik OK untuk Mulai', title='Perhatian !')
            
    #=====================This Segment For Image Processing===============================================
            # Download Data Set Gambar
            def download(filename, source='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'):
                print("Downloading %s" % filename)
                urlretrieve(source + filename, filename)

            # Jika Tidak ada File dataset download(), jika ada maka baca dataset
            def load_fashion_mnist_images(filename):
                if not os.path.exists(filename):
                    download(filename)
                with gzip.open(filename, 'rb') as f:
                    data = np.frombuffer(f.read(), np.uint8, offset=16)
                data = data.reshape(-1,784)
                return data

            def load_fashion_mnist_labels(filename):
                if not os.path.exists(filename):
                    download(filename)
                with gzip.open(filename, 'rb') as f:
                    data = np.frombuffer(f.read(), np.uint8, offset=8)
                return data

            ## Load dataset
            train_data = load_fashion_mnist_images('train-images-idx3-ubyte.gz')
            train_labels = load_fashion_mnist_labels('train-labels-idx1-ubyte.gz')
            ## Load testing set
            test_data = load_fashion_mnist_images('t10k-images-idx3-ubyte.gz')
            test_labels = load_fashion_mnist_labels('t10k-labels-idx1-ubyte.gz')
            print(train_data.shape)
            # (60000, 784) ## 60k 28x28 images
            print(test_data.shape)
            # (10000, 784) ## 10k 2bx28 images
            print(np.max(train_data))

            products = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
            print(len(products))
            ## Mendefinisikan fungsi yang menampilkan gambar yang diberikan dalam vektor
            def show_image(x, label):
                plt.imshow(x.reshape((28,28)), cmap=plt.cm.gray)
                plt.title(products[label], size=15)
                plt.axis('off')

            plt.figure(figsize=(20,20))
            for i in range(100):
                plt.subplot(10, 10, i+1)
                show_image(test_data[i,:], test_labels[i])
            plt.tight_layout()
            plt.savefig('images/original/original.png')
            #plt.show()


            # normalize
            X_train = np.zeros(train_data.shape)
            for i in range(train_data.shape[0]):
                X_train[i,:] = train_data[i,:] / np.max(train_data[i,:])
            X_test = np.zeros(test_data.shape)
            for i in range(test_data.shape[0]):
                X_test[i,:] = test_data[i,:] / np.max(test_data[i,:])
                
            global VAE    
            class VAE(nn.Module):
                def __init__(self):
                    super(VAE, self).__init__()

                    self.fc1 = nn.Linear(n*n, 512)
                    self.fc21 = nn.Linear(512, 32) # mu        # must change to (512, 2) if you want a 2D VAE 
                    self.fc22 = nn.Linear(512, 32) # sigma     # must change to (512, 2) if you want a 2D VAE 
                    self.fc3 = nn.Linear(32, 512)
                    self.fc4 = nn.Linear(512, n*n)

                def encode(self, x):
                    h1 = F.relu(self.fc1(x))
                    return self.fc21(h1), self.fc22(h1)

                def reparameterize(self, mu, logvar):
                    std = torch.exp(0.5*logvar)
                    eps = torch.randn_like(std)
                    return mu + eps*std

                def decode(self, z):
                    h3 = F.relu(self.fc3(z))
                    return torch.sigmoid(self.fc4(h3))

                def forward(self, x):
                    mu, logvar = self.encode(x.view(-1, n*n))
                    z = self.reparameterize(mu, logvar)
                    return self.decode(z), mu, logvar    

            torch.manual_seed(1)

            cuda = torch.cuda.is_available()
            batch_size = 512 #128
            log_interval = 20
            epochs = 20
            n = 28

            device = torch.device("cuda" if cuda else "cpu")

            kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

            train_loader = DataLoader(np.reshape(X_train, (-1, 1, n, n)).astype(np.float32), batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(np.reshape(X_test, (-1, 1, n, n)).astype(np.float32), batch_size=batch_size, shuffle=True)

            model = VAE().to(device)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)


            # Rekonstruksi + rugi divergensi KL dijumlahkan untuk semua elemen dan batch
            def loss_function(recon_x, x, mu, logvar):
                BCE = F.binary_cross_entropy(recon_x, x.view(-1, n*n), reduction='sum')
                KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

                return BCE + KLD


            def train(epoch):
                model.train()
                batch_idx = 0
                train_loss = 0
                for data in train_loader:
                    data = data.to(device)
                    optimizer.zero_grad()
                    recon_batch, mu, logvar = model(data)
                    loss = loss_function(recon_batch, data, mu, logvar)
                    loss.backward()
                    train_loss += loss.item()
                    optimizer.step()
                    if batch_idx % log_interval == 0:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch, batch_idx * len(data), len(train_loader.dataset),
                            100. * batch_idx / len(train_loader),
                            loss.item() / len(data)))
                    batch_idx += 1

                print('====> Epoch: {} Average loss: {:.4f}'.format(
                      epoch, train_loss / len(train_loader.dataset)))
                
            def test(epoch):
                model.eval()
                losses = []
                i = 0
                test_loss = 0
                with torch.no_grad():
                    for data in test_loader:
                        data = data.to(device)
                        recon_batch, mu, logvar = model(data)
                        test_loss += loss_function(recon_batch, data, mu, logvar).item()
                        if i == 0:
                            N = min(data.size(0), 8)
                            comparison = torch.cat([data[:N],
                                                  recon_batch.view(batch_size, 1, n, n)[:N]])
                            save_image(comparison.cpu(),
                                     'images/reconstruction/reconstruction_' + str(epoch) + '.png', nrow=N)
                            i += 1

                test_loss /= len(test_loader.dataset)
                print('====> Test set loss: {:.4f}'.format(test_loss))

            for epoch in range(1, epochs + 1):
                train(epoch)
                test(epoch)
                with torch.no_grad():
                    sample = torch.randn(64, 32).to(device)
                    sample = model.decode(sample).cpu()
                    save_image(sample.view(64, 1, n, n),
                               'images/sample/sample_' + str(epoch) + '.png')
            torch.save(model, 'models/vae.pth')

            with torch.no_grad():
                mu, _ = model.encode(torch.from_numpy(X_test).float().to(device))
            mu = mu.cpu().numpy()
              
            plt.figure(figsize=(15, 10)) 
            plt.scatter(mu[:, 0], mu[:, 1], c=test_labels, cmap='jet'), plt.colorbar()
            plt.xlabel({i:products[i] for i in range(len(products))}, fontsize=15)
            plt.savefig('images/graph/grafik.png')
            #plt.show()
            
            
            #=================2D VAE=======================================
            def load_fashion_mnist_labels(filename):
                if not os.path.exists(filename):
                    download(filename)
                with gzip.open(filename, 'rb') as f:
                    data = np.frombuffer(f.read(), np.uint8, offset=8)
                return data

            ## Load dataset
            train_data = load_fashion_mnist_images('train-images-idx3-ubyte.gz')
            train_labels = load_fashion_mnist_labels('train-labels-idx1-ubyte.gz')
            ## Load testing set
            test_data = load_fashion_mnist_images('t10k-images-idx3-ubyte.gz')
            test_labels = load_fashion_mnist_labels('t10k-labels-idx1-ubyte.gz')
            print(train_data.shape)
            # (60000, 784) ## 60k 28x28 images
            print(test_data.shape)
            # (10000, 784) ## 10k 2bx28 images
            print(np.max(train_data))

            products = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
            print(len(products))
            ## Mendefinisikan fungsi yang menampilkan gambar yang diberikan dalam vektor
            def show_image(x, label):
                plt.imshow(x.reshape((28,28)), cmap=plt.cm.gray)
                plt.title(products[label], size=15)
                plt.axis('off')

            # normalize
            X_train = np.zeros(train_data.shape)
            for i in range(train_data.shape[0]):
                X_train[i,:] = train_data[i,:] / np.max(train_data[i,:])
            X_test = np.zeros(test_data.shape)
            for i in range(test_data.shape[0]):
                X_test[i,:] = test_data[i,:] / np.max(test_data[i,:])
                    
            class VAE(nn.Module):
                def __init__(self):
                    super(VAE, self).__init__()

                    self.fc1 = nn.Linear(n*n, 512)
                    self.fc21 = nn.Linear(512, 2) # mu        # must change to (512, 2) if you want a 2D VAE 
                    self.fc22 = nn.Linear(512, 2) # sigma     # must change to (512, 2) if you want a 2D VAE 
                    self.fc3 = nn.Linear(2, 512)
                    self.fc4 = nn.Linear(512, n*n)

                def encode(self, x):
                    h1 = F.relu(self.fc1(x))
                    return self.fc21(h1), self.fc22(h1)

                def reparameterize(self, mu, logvar):
                    std = torch.exp(0.5*logvar)
                    eps = torch.randn_like(std)
                    return mu + eps*std

                def decode(self, z):
                    h3 = F.relu(self.fc3(z))
                    return torch.sigmoid(self.fc4(h3))

                def forward(self, x):
                    mu, logvar = self.encode(x.view(-1, n*n))
                    z = self.reparameterize(mu, logvar)
                    return self.decode(z), mu, logvar    

            torch.manual_seed(1)
            cuda = torch.cuda.is_available()
            batch_size = 512 #128
            log_interval = 20
            epochs = 20
            n = 28
            device = torch.device("cuda" if cuda else "cpu")
            kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
            train_loader = DataLoader(np.reshape(X_train, (-1, 1, n, n)).astype(np.float32), batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(np.reshape(X_test, (-1, 1, n, n)).astype(np.float32), batch_size=batch_size, shuffle=True)
            model = VAE().to(device)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)


            # Rekonstruksi + rugi divergensi KL dijumlahkan untuk semua elemen dan batch
            def loss_function(recon_x, x, mu, logvar):
                BCE = F.binary_cross_entropy(recon_x, x.view(-1, n*n), reduction='sum')
                KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                return BCE + KLD
            
            def train(epoch):
                model.train()
                batch_idx = 0
                train_loss = 0
                for data in train_loader:
                    data = data.to(device)
                    optimizer.zero_grad()
                    recon_batch, mu, logvar = model(data)
                    loss = loss_function(recon_batch, data, mu, logvar)
                    loss.backward()
                    train_loss += loss.item()
                    optimizer.step()
                    if batch_idx % log_interval == 0:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch, batch_idx * len(data), len(train_loader.dataset),
                            100. * batch_idx / len(train_loader),
                            loss.item() / len(data)))
                    batch_idx += 1
                print('====> Epoch: {} Average loss: {:.4f}'.format(
                      epoch, train_loss / len(train_loader.dataset)))
            
            def test(epoch):
                model.eval()
                losses = []
                i = 0
                test_loss = 0
                with torch.no_grad():
                    for data in test_loader:
                        data = data.to(device)
                        recon_batch, mu, logvar = model(data)
                        test_loss += loss_function(recon_batch, data, mu, logvar).item()
                        if i == 0:
                            N = min(data.size(0), 8)
                            comparison = torch.cat([data[:N],
                                                  recon_batch.view(batch_size, 1, n, n)[:N]])
                            #save_image(comparison.cpu(),'images/2d_reconstruction/reconstruction_' + str(epoch) + '.png', nrow=N)
                            i += 1

                test_loss /= len(test_loader.dataset)
                print('====> Test set loss: {:.4f}'.format(test_loss))
            
            for epoch in range(1, epochs + 1):
                train(epoch)
                test(epoch)
                with torch.no_grad():
                    sample = torch.randn(64, 2).to(device)
                    sample = model.decode(sample).cpu()
                    #save_image(sample.view(64, 1, n, n),'images/2d_sample/sample_' + str(epoch) + '.png')
            torch.save(model, 'models/vae.pth')
            with torch.no_grad():
                mu, _ = model.encode(torch.from_numpy(X_test).float().to(device))
            mu = mu.cpu().numpy()    
            
            plt.figure(figsize=(5,5))
            plt.subplots_adjust(0,0,1,0.95,0.05,0.05)
            for i in range(64):
                z_mu = np.random.normal(0,1,2).reshape(1,2)
                x = model.decode(torch.from_numpy(z_mu).float().to(device)).cpu().detach().numpy()
                plt.subplot(8,8,i+1)
                plt.imshow(x.reshape(n, n))
                plt.axis('off')
            plt.suptitle('', size=20)
            plt.savefig('images/2d/2d_generated.png')
            #plt.show()   
            plt.figure(figsize=(15, 10)) 
            plt.scatter(mu[:, 0], mu[:, 1], c=test_labels, cmap='jet'), plt.colorbar()
            plt.xlabel({i:products[i] for i in range(len(products))}, fontsize=15)
            plt.savefig('images/2d/2d_grafik.png')
            
            nx = ny = 20
            x_values = np.linspace(-3, 3, nx)
            y_values = np.linspace(-3, 3, ny)

            with torch.no_grad():
                canvas = np.empty((n*ny, n*nx))
                for i, yi in enumerate(x_values):
                    for j, xi in enumerate(y_values):
                        z_mu = np.array([[xi, yi]*100]).reshape(100,2)
                        x_mean = model.decode(torch.from_numpy(z_mu).float().to(device)).cpu().numpy()
                        canvas[(nx-i-1)*n:(nx-i)*n, j*n:(j+1)*n] = x_mean[0].reshape(n, n)

            plt.figure(figsize=(8, 10))  
            plt.gray()
            Xi, Yi = np.meshgrid(x_values, y_values)
            plt.imshow(canvas, origin="upper")
            plt.tight_layout()
            plt.savefig('images/2d/2d_latent_space.png')
            
            winsound.PlaySound("ButtonClick.wav", 1) if event != sg.TIMEOUT_KEY else None
            sg.popup('Gambar Telah Diproses, Anda Dapat Melihat Hasilnya Pada GUI','','Klik OK untuk Melanjutkan', title='Berhasil !!!')
            
            #===================END OF IMAGE PROCESSING =======================
        
        elif event == 'GA':
            window.close()
            winsound.PlaySound("ButtonClick.wav", 1) if event != sg.TIMEOUT_KEY else None
            gambar_asli = "images/original/original.png"
            # Resize PNG file to size (300, 300)
            size = (500, 500)
            im = Image.open(gambar_asli)
            im = im.resize(size, resample=Image.BICUBIC)

            

            layout = [
                [sg.Image(size=(500, 500), key='-IMAGE-')],
            ]
            window = sg.Window('Gambar Original', layout, margins=(0, 0), finalize=True)

            # Convert im to ImageTk.PhotoImage after window finalized
            image = ImageTk.PhotoImage(image=im)

            # update image in sg.Image
            window['-IMAGE-'].update(data=image)

            while True:
                event, values = window.read()
                if event == sg.WIN_CLOSED:
                    break
            
            window.close()
            start()
                
        
        
        elif event == 'GRV':
            window.close()
            winsound.PlaySound("ButtonClick.wav", 1) if event != sg.TIMEOUT_KEY else None
            # Get Folder
            folder = "images/reconstruction" #Nama Folder
            if folder is None:
                sg.popup_cancel('Cancelling')
                return

            # get list 
            png_files = [os.path.join(folder, f) 
                         for f in os.listdir(folder) 
                         if f.lower().endswith('.png')]
            filenames_only = [f for f in os.listdir(folder) 
                              if f.lower().endswith('.png')]

            if len(png_files) == 0:
                sg.popup('Tidak ada Gambar !')
                return

            # menu layout
            menu = [['File', ['Exit']], ['Help', ['About', ]]]
            


            # menampikan window dan didefinisikan
            col = [[sg.Text(png_files[0], 
                            size=(30, 3), 
                            key='-FILENAME-')],
                   [sg.Image(filename=png_files[0],
                             key='-IMAGE-')],
                   [sg.Button('Next', size=(8, 2)), 
                    sg.Button('Prev', size=(8, 2)),
                    sg.Text('File 1 of {}'.format(len(png_files)), 
                            size=(15, 1), 
                            key='-FILENUM-')]]

            col_files = [[sg.Listbox(values=filenames_only, 
                                     size=(30, 20), 
                                     key='-LISTBOX-', 
                                     enable_events=True)],
                         [sg.Text('Hasil Reconstruksi Gambar')],
                         [sg.Text('Menggunakan VAE')]]

            layout = [[sg.Menu(menu)], [sg.Col(col_files), sg.Col(col)]]

            window = sg.Window('HASIL REKONSTRUKSI', layout, return_keyboard_events=True, use_default_focus=False)

            # loop gambar
            filenum, filename = 0, png_files[0]
            while True:

                event, values = window.read()
                # --------------------- Button ---------------------
                if event == sg.WIN_CLOSED:
                    break
                elif event in ('Next', 'MouseWheel:Down', 'Down:40', 'Next:34') and filenum < len(png_files)-1:
                    filenum += 1
                    filename = os.path.join(folder, filenames_only[filenum])
                    window['-LISTBOX-'].update(set_to_index=filenum, scroll_to_index=filenum)
                elif event in ('Prev', 'MouseWheel:Up', 'Up:38', 'Prior:33') and filenum > 0:
                    filenum -= 1
                    filename = os.path.join(folder, filenames_only[filenum])
                    window['-LISTBOX-'].update(set_to_index=filenum, scroll_to_index=filenum)
                elif event == 'Exit':
                    break
                elif event == '-LISTBOX-':
                    filename = os.path.join(folder, values['-LISTBOX-'][0])
                    filenum = png_files.index(filename)
                # ----------------- Pilihan Menu -----------------
                
                if event == 'About':
                    sg.popup('UAS PENGOLAHAN CITRA',
                             'NAMA : GALIH AJI PAMBUDI','NIM : 3332180058',
                             '',
                             'Using a variational autoencoder to reconstruct and generate images')
                    filenum = 0

                # update window dengan gambar baru
                window['-IMAGE-'].update(filename=filename)
                # update window dengan namafile
                window['-FILENAME-'].update(filename)
                # update tampilan halaman
                window['-FILENUM-'].update('File {} of {}'.format(filenum + 1, len(png_files)))

            window.close() 
            start()
            
            
        
        elif event == 'GGV':
            window.close()
            winsound.PlaySound("ButtonClick.wav", 1) if event != sg.TIMEOUT_KEY else None
            # Get Folder
            folder = "images/sample" #Nama Folder
            if folder is None:
                sg.popup_cancel('Cancelling')
                return

            # get list 
            png_files = [os.path.join(folder, f) 
                         for f in os.listdir(folder) 
                         if f.lower().endswith('.png')]
            filenames_only = [f for f in os.listdir(folder) 
                              if f.lower().endswith('.png')]

            if len(png_files) == 0:
                sg.popup('Tidak ada Gambar !')
                return

            # menu layout
            menu = [['File', ['Exit']], ['Help', ['About', ]]]
            


            # menampikan window dan didefinisikan
            col = [[sg.Text(png_files[0], 
                            size=(30, 3), 
                            key='-FILENAME-')],
                   [sg.Image(filename=png_files[0],
                             key='-IMAGE-')],
                   [sg.Button('Next', size=(8, 2)), 
                    sg.Button('Prev', size=(8, 2)),
                    sg.Text('File 1 of {}'.format(len(png_files)), 
                            size=(15, 1), 
                            key='-FILENUM-')]]

            col_files = [[sg.Listbox(values=filenames_only, 
                                     size=(30, 20), 
                                     key='-LISTBOX-', 
                                     enable_events=True)],
                         [sg.Text('Hasil Generated Gambar')],
                         [sg.Text('Menggunakan VAE')]]

            layout = [[sg.Menu(menu)], [sg.Col(col_files), sg.Col(col)]]

            window = sg.Window('HASIL GENERATED GAMBAR', layout, return_keyboard_events=True, use_default_focus=False)

            # loop gambar
            filenum, filename = 0, png_files[0]
            while True:

                event, values = window.read()
                # --------------------- Button ---------------------
                if event == sg.WIN_CLOSED:
                    break
                elif event in ('Next', 'MouseWheel:Down', 'Down:40', 'Next:34') and filenum < len(png_files)-1:
                    filenum += 1
                    filename = os.path.join(folder, filenames_only[filenum])
                    window['-LISTBOX-'].update(set_to_index=filenum, scroll_to_index=filenum)
                elif event in ('Prev', 'MouseWheel:Up', 'Up:38', 'Prior:33') and filenum > 0:
                    filenum -= 1
                    filename = os.path.join(folder, filenames_only[filenum])
                    window['-LISTBOX-'].update(set_to_index=filenum, scroll_to_index=filenum)
                elif event == 'Exit':
                    break
                elif event == '-LISTBOX-':
                    filename = os.path.join(folder, values['-LISTBOX-'][0])
                    filenum = png_files.index(filename)
                # ----------------- Pilihan Menu -----------------
                
                if event == 'About':
                    sg.popup('UAS PENGOLAHAN CITRA',
                             'NAMA : GALIH AJI PAMBUDI','NIM : 3332180058',
                             '',
                             'Using a variational autoencoder to reconstruct and generate images')
                    filenum = 0

                # update window dengan gambar baru
                window['-IMAGE-'].update(filename=filename)
                # update window dengan namafile
                window['-FILENAME-'].update(filename)
                # update tampilan halaman
                window['-FILENUM-'].update('File {} of {}'.format(filenum + 1, len(png_files)))

            window.close() 
            start()
            
            
        
        elif event == 'grafik':
            winsound.PlaySound("ButtonClick.wav", 1) if event != sg.TIMEOUT_KEY else None
            window.close()
            winsound.PlaySound("ButtonClick.wav", 1) if event != sg.TIMEOUT_KEY else None
            gambar_asli = "images/graph/grafik.png"
            # Resize PNG file to size (300, 300)
            size = (700, 500)
            im = Image.open(gambar_asli)
            im = im.resize(size, resample=Image.BICUBIC)

            layout = [
                [sg.Image(size=(700, 500), key='-IMAGE-')],
            ]
            window = sg.Window('GRAFIK VAE', layout, margins=(0, 0), finalize=True)

            # Convert im to ImageTk.PhotoImage after window finalized
            image = ImageTk.PhotoImage(image=im)

            # update image in sg.Image
            window['-IMAGE-'].update(data=image)

            while True:
                event, values = window.read()
                if event == sg.WIN_CLOSED:
                    break
            
            window.close()
            start()
            
        elif event == 'G2V':
            window.close()
            winsound.PlaySound("ButtonClick.wav", 1) if event != sg.TIMEOUT_KEY else None
            gambar_asli = "images/2d/2d_generated.png"
            # Resize PNG file to size (300, 300)
            size = (500, 500)
            im = Image.open(gambar_asli)
            im = im.resize(size, resample=Image.BICUBIC)

            

            layout = [
                [sg.Image(size=(500, 500), key='-IMAGE-')],
            ]
            window = sg.Window('Gambar dibuat oleh 2D VAE', layout, margins=(0, 0), finalize=True)

            # Convert im to ImageTk.PhotoImage after window finalized
            image = ImageTk.PhotoImage(image=im)

            # update image in sg.Image
            window['-IMAGE-'].update(data=image)

            while True:
                event, values = window.read()
                if event == sg.WIN_CLOSED:
                    break
            
            window.close()
            start()
            
        
            
        elif event == 'L2V':
            window.close()
            winsound.PlaySound("ButtonClick.wav", 1) if event != sg.TIMEOUT_KEY else None
            gambar_asli = "images/2d/2d_latent_space.png"
            # Resize PNG file to size (300, 300)
            size = (500, 500)
            im = Image.open(gambar_asli)
            im = im.resize(size, resample=Image.BICUBIC)

            

            layout = [
                [sg.Image(size=(500, 500), key='-IMAGE-')],
            ]
            window = sg.Window('Visualisasi Gambar dari Latent Space', layout, margins=(0, 0), finalize=True)

            # Convert im to ImageTk.PhotoImage after window finalized
            image = ImageTk.PhotoImage(image=im)

            # update image in sg.Image
            window['-IMAGE-'].update(data=image)

            while True:
                event, values = window.read()
                if event == sg.WIN_CLOSED:
                    break
            
            window.close()
            start()
            
        elif event == 'G2A':
            window.close()
            winsound.PlaySound("ButtonClick.wav", 1) if event != sg.TIMEOUT_KEY else None
            gambar_asli = "images/2d/2d_grafik.png"
            # Resize PNG file to size (300, 300)
            size = (700, 500)
            im = Image.open(gambar_asli)
            im = im.resize(size, resample=Image.BICUBIC)

            

            layout = [
                [sg.Image(size=(700, 500), key='-IMAGE-')],
            ]
            window = sg.Window('GRAFIK 2D VAE', layout, margins=(0, 0), finalize=True)

            # Convert im to ImageTk.PhotoImage after window finalized
            image = ImageTk.PhotoImage(image=im)

            # update image in sg.Image
            window['-IMAGE-'].update(data=image)

            while True:
                event, values = window.read()
                if event == sg.WIN_CLOSED:
                    break
            
            window.close()
            start()
    
  
    window.close()
start()
