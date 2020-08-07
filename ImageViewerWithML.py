#Unsupervised learning using ResNet50 and Kmeans
#https://towardsdatascience.com/image-clustering-using-transfer-learning-df5862779571
#UI added
#OTIS 14 Feb 2020 - Quentin BEZAULT

#For UI
from PyQt5.QtCore import QDir, Qt, pyqtSlot
from PyQt5.QtGui import QImage, QPainter, QPalette, QPixmap
from PyQt5.QtWidgets import (QAction, QApplication, QFileDialog, QLabel,
        QMainWindow, QMenu, QMessageBox, QScrollArea, QSizePolicy, QLineEdit, QPushButton,
        QTreeWidget, QTreeWidgetItem, QWidget)
import os

#For Machine Learning
from pathlib import Path
import cv2
from sklearn.cluster import KMeans
import numpy as np
import glob
from keras.applications.resnet50 import ResNet50
from keras.models import Sequential
from keras.applications.resnet50 import preprocess_input
import time
import csv

#For data visualization
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import seaborn as sns

#Main window : Image Viewer

class ImageViewer(QMainWindow):
    def __init__(self):
        super(ImageViewer, self).__init__()
        
        self.tree = QTreeWidget()

        self.scaleFactor = 0.0

        self.imageLabel = QLabel()
        self.imageLabel.setBackgroundRole(QPalette.Base)
        self.imageLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.imageLabel.setScaledContents(True)

        self.scrollArea = QScrollArea()
        self.scrollArea.setBackgroundRole(QPalette.Dark)
        self.scrollArea.setWidget(self.imageLabel)
        self.setCentralWidget(self.scrollArea)

        self.createActions()
        self.createMenus()
        self.createTextbox()
        self.createNextbutton()
        self.createShowButton()
        self.createUpdatebutton()
        self.createSaveFeaturesButton()
        self.createLoadFeaturesButton()
        self.createShowPreviewButton()

        self.setWindowTitle("AI Image Classifier")

        self.resize(900, 600)

        self.tree.itemSelectionChanged.connect(lambda: self.loadAllMessages(self.tree))

        self.InitMachineLearning()

     
    def open(self):
        self.fileName, _ = QFileDialog.getOpenFileName(self, "Open File", QDir.currentPath())

        image = QImage(self.fileName)
        if image.isNull():
            QMessageBox.information(self, "Image Viewer",
                    "Cannot load %s." % self.fileName)
            return

        self.imageLabel.setPixmap(QPixmap.fromImage(image))
        self.scaleFactor = 1.0

        self.fitToWindowAct.setEnabled(True)
        self.updateActions()

        if not self.fitToWindowAct.isChecked():
            self.imageLabel.adjustSize()

    def showimage(self):
        getSelected = self.tree.selectedItems()
        if getSelected:
            baseNode = getSelected[0]
            getChildNode = baseNode.text(0)
            self.image_path = Path(self.fileName).parent.joinpath(Path(getChildNode))
            #print(getChildNode) #Affiche le nom du fichier image xxxx.jpg

            image = QImage(str(self.image_path))

            self.imageLabel.setPixmap(QPixmap.fromImage(image))
            self.scaleFactor = 1.0

            self.fitToWindowAct.setEnabled(True)
            self.updateActions()

            if not self.fitToWindowAct.isChecked():
                self.imageLabel.adjustSize()

    def zoomIn(self):
        self.scaleImage(1.25)

    def zoomOut(self):
        self.scaleImage(0.8)

    def normalSize(self):
        self.imageLabel.adjustSize()
        self.scaleFactor = 1.0

    def fitToWindow(self):
        fitToWindow = self.fitToWindowAct.isChecked()
        self.scrollArea.setWidgetResizable(fitToWindow)
        if not fitToWindow:
            self.normalSize()

        self.updateActions()

    def about(self):
        QMessageBox.about(self, "Pictogram detection for machine learning",
                "<p>This software analyzes pictures to sort them in a defined number of clusters</p> ")

    def createActions(self):
        self.openAct = QAction("&Open...", self, shortcut="Ctrl+O",
                triggered=self.open)

        self.exitAct = QAction("E&xit", self, shortcut="Ctrl+Q",
                triggered=self.close)

        self.zoomInAct = QAction("Zoom &In (25%)", self, shortcut="Ctrl++",
                enabled=False, triggered=self.zoomIn)

        self.zoomOutAct = QAction("Zoom &Out (25%)", self, shortcut="Ctrl+-",
                enabled=False, triggered=self.zoomOut)

        self.normalSizeAct = QAction("&Normal Size", self, shortcut="Ctrl+S",
                enabled=False, triggered=self.normalSize)

        self.fitToWindowAct = QAction("&Fit to Window", self, enabled=False,
                checkable=True, shortcut="Ctrl+F", triggered=self.fitToWindow)

        self.aboutAct = QAction("&About", self, triggered=self.about)

        self.aboutQtAct = QAction("About &Qt", self,
                triggered=QApplication.instance().aboutQt)

    def createMenus(self):
        self.fileMenu = QMenu("&File", self)
        self.fileMenu.addAction(self.openAct)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.exitAct)

        self.viewMenu = QMenu("&View", self)
        self.viewMenu.addAction(self.zoomInAct)
        self.viewMenu.addAction(self.zoomOutAct)
        self.viewMenu.addAction(self.normalSizeAct)
        self.viewMenu.addSeparator()
        self.viewMenu.addAction(self.fitToWindowAct)

        self.helpMenu = QMenu("&Help", self)
        self.helpMenu.addAction(self.aboutAct)
        self.helpMenu.addAction(self.aboutQtAct)

        self.menuBar().addMenu(self.fileMenu)
        self.menuBar().addMenu(self.viewMenu)
        self.menuBar().addMenu(self.helpMenu)

    def updateActions(self):
        self.zoomInAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.zoomOutAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.normalSizeAct.setEnabled(not self.fitToWindowAct.isChecked())

    def scaleImage(self, factor):
        self.scaleFactor *= factor
        self.imageLabel.resize(self.scaleFactor * self.imageLabel.pixmap().size())

        self.adjustScrollBar(self.scrollArea.horizontalScrollBar(), factor)
        self.adjustScrollBar(self.scrollArea.verticalScrollBar(), factor)

        self.zoomInAct.setEnabled(self.scaleFactor < 5.0)
        self.zoomOutAct.setEnabled(self.scaleFactor > 0.2)

    def adjustScrollBar(self, scrollBar, factor):
        scrollBar.setValue(int(factor * scrollBar.value()
                                + ((factor - 1) * scrollBar.pageStep()/2)))

    def createTextbox(self):         
        self.textbox = QLineEdit(self)
        self.textbox.move(650, 560)
        self.textbox.resize(50,20)

    def createShowButton(self):         
        nbtn = QPushButton('Show', self)
        nbtn.clicked.connect(self.showimage)
        nbtn.resize(nbtn.sizeHint())
        nbtn.move(700, 540)

    def createNextbutton(self):         
        nbtn = QPushButton('Analyze', self)
        nbtn.clicked.connect(self.analyzepictures)
        nbtn.resize(nbtn.sizeHint())
        nbtn.move(700, 580)

    def createUpdatebutton(self):         
        nbtn = QPushButton('Update', self)
        nbtn.clicked.connect(self.updateclusters)
        nbtn.resize(nbtn.sizeHint())
        nbtn.move(700, 560)

    def createSaveFeaturesButton(self):         
        nbtn = QPushButton('Save', self)
        nbtn.clicked.connect(self.savearray)
        nbtn.resize(nbtn.sizeHint())
        nbtn.move(800, 540)

    def createLoadFeaturesButton(self):         
        nbtn = QPushButton('Load', self)
        nbtn.clicked.connect(self.loadarray)
        nbtn.resize(nbtn.sizeHint())
        nbtn.move(800, 560)

    def createShowPreviewButton(self):         
        nbtn = QPushButton('Preview', self)
        nbtn.clicked.connect(self.on_ShowPreviewButton_Clicked)
        nbtn.resize(nbtn.sizeHint())
        nbtn.move(800, 580)
    
    def on_ShowPreviewButton_Clicked(self):
        ClusterPreviewMatrix = ImageMatrix()
        ClusterPreviewMatrix.show()

    def InitMachineLearning(self):
        #Initialization steps for machine learning
        resnet_weights_path = 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

        self.my_new_model = Sequential()
        self.my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))

        # Say not to train first layer (ResNet) model. It is already trained
        self.my_new_model.layers[0].trainable = False
        print('ML Model initiated')

    @pyqtSlot()
    def analyzepictures(self):
        start = time.time()
        self.array,self.path_list = self.extract_vector(str(Path(self.fileName).parent) +'\*' +str(Path(self.fileName).suffix))
        end = time.time()
        print('Extract features : 100 %')       
        print('Photos analyzed in' + "{:10.2f}".format(end-start) + ' s')
        
        self.updateclusters()


    def savearray(self):
        np.save('array.npy', self.array)
        np.array(self.path_list)
        np.save('paths.npy', self.path_list)  
        np.save('filename.npy', self.fileName)
        np.save('kmeans_labels.npy', self.kmeans.labels_)
        np.save('kmeans_nclusters.npy', self.kmeans.n_clusters)       
        print('Saved')

    def loadarray(self):
        self.array = np.load('array.npy')
        self.path_list = np.load('paths.npy')
        self.fileName = np.load('filename.npy')
        self.fileName = str(self.fileName)
        self.kmeans.labels_ = np.load('kmeans_labels.npy')
        self.kmeans.n_clusters = np.load('kmeans_nclusters.npy')
        self.createTree(self.kmeans.labels_,self.path_list)
        print('Loaded')

    # def tsne(self):
    #     time_start = time.time()
    #     tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    #     tsne_results = tsne.fit_transform(data_subset)
    #     print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    #     df_subset['tsne-2d-one'] = tsne_results[:,0]
    #     df_subset['tsne-2d-two'] = tsne_results[:,1]
    #     plt.figure(figsize=(16,10))
    #     sns.scatterplot(
    #         x="tsne-2d-one", y="tsne-2d-two",
    #         hue="y",
    #         palette=sns.color_palette("hls", 10),
    #         data=df_subset,
    #         legend="full",
    #         alpha=0.3
    #     )

    def updateclusters(self): #Same as analyzepictures but without extract vector call
        start = time.time()
        self.kmeans = KMeans(n_clusters=int(self.textbox.text()), random_state=0, verbose=1).fit(self.array)
        self.createTree(self.kmeans.labels_,self.path_list)
        end = time.time()
        print('Clusters computed in' + "{:10.2f}".format(end-start) + ' s')

    def populatetree(self,labels,path_list):
        photocountarray = []
        labelslist = list(labels)
        #Count photos in each cluster
        for i in range(int(self.kmeans.n_clusters)):
            photocountarray.append(labelslist.count(i))

        #Sort clusters by ascending photos count
        argsorted = np.argsort(photocountarray)

        for i in range(len(labels)):
            newlabel = np.where(argsorted == labels[i])
            labels[i]= newlabel[0]

        #Populate photo tree
        for i in range(int(self.kmeans.n_clusters)):
            photo_count = 0
            parent = QTreeWidgetItem(self.tree)
            for j in range(len(labels)):
                if labels[j]==i :
                    child = QTreeWidgetItem(parent)
                    child.setText(0, str(path_list[j]))
                    photo_count += 1
            parent.setText(0, "Cluster {}".format(i+1) + ' - ' + str(photo_count) + ' photos')

    def createTree(self,labels,path_list):
        self.tree = QTreeWidget(self)
        self.populatetree(labels,path_list)    
        self.tree.move(600, 20)
        self.tree.resize(300,500)
        self.tree.show()


    #Function for machine learning
    def extract_vector(self,path):
        resnet_feature_list = []
        path_list= []
        counter_progress = 0

        for im in glob.glob(path):
            photofilename = Path(im)
            path_list.append(photofilename.name)
            im = cv2.imread(im)
            #im = cv2.resize(im,(448,448))
            img = preprocess_input(np.expand_dims(im.copy(), axis=0))
            resnet_feature = self.my_new_model.predict(img)
            resnet_feature_np = np.array(resnet_feature)
            resnet_feature_list.append(resnet_feature_np.flatten())

            #Show progress
            counter_progress = counter_progress + 1
            progress = 100*counter_progress / len(glob.glob(path))
            print ('Extract features :'"{:10.2f}".format(progress) + ' %', end="\r")
            
        return np.array(resnet_feature_list),path_list    

class ImageMatrix(QMainWindow):
    def __init__(self):
        super(ImageMatrix, self).__init__()
        self.scaleFactor = 0.0

        self.imageLabel = QLabel()
        self.imageLabel.setBackgroundRole(QPalette.Base)
        self.imageLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.imageLabel.setScaledContents(True)

        self.scrollArea = QScrollArea()
        self.scrollArea.setBackgroundRole(QPalette.Dark)
        self.scrollArea.setWidget(self.imageLabel)
        self.setCentralWidget(self.scrollArea)

        self.setWindowTitle("Clusters preview")

        self.resize(900, 600)

if __name__ == '__main__':

    import sys

    app = QApplication(sys.argv)
    imageViewer = ImageViewer()
    imageViewer.show()
    sys.exit(app.exec_())