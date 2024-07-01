'''rst@ptb 240429'''
'''field viewer class for visualizing B0 maps recorded with cosi measure'''

from PyQt5 import QtWidgets, uic
import os
from time import sleep # for dev only

from utils import Plotter  # an EMRE module for plotting
import b0  # class of b0, makes an object of a b0. Can import from file. Attributes: see shimming_david repo

from PyQt5 import QtWidgets
import numpy as np

from utils import shimming_magnet


class field_viewer_gui(QtWidgets.QMainWindow):
    '''the field viewer window.'''
    
    b0map = b0.b0  # instance of b0 scan, make an object.

    workingFolder = r"./dummies/b0_maps"  # where the openfiledialog opens

    def __init__(self):
        super(field_viewer_gui, self).__init__()  # Call the inherited classes __init__ method

        uic.loadUi('gui/field_viewer.ui', self)  # Load the .ui file
        self.show()  # Show the GUI

        # working folder
        self.b0Path = None

        # binding methods to buttons:
        self.save_button.clicked.connect(self.save_rotated_path_in_a_csv_file)  # Remember to code the method in the class.
        #self.load_button.clicked.connect(self.load_b0)  # old import, for COSI v.1
        self.load_csv_button.clicked.connect(self.load_csv)  # Remember to code the method in the class.
        self.show_2d_slice_btn.clicked.connect(self.plot_B0M_slice_2d)
        self.export_for_Tom_btn.clicked.connect(self.export_separately)
        
        # data transformation buttons
        self.SPH_button.clicked.connect(self.fit_sph)
        
        # shimming buttons
        self.get_shim_positions_btn.clicked.connect(self.get_shim_positions)
        self.save_rings_button.clicked.connect(self.save_shim_magnets_in_rings)
        self.load_rings_button.clicked.connect(self.load_shim_magnets_in_rings)

        # --- adding the plotter: ---
        # B0M plotter:
        plotterWidgetFound = self.findChild(QtWidgets.QWidget, 'plotter_widget')
        self.plotterWGT = Plotter.Plotter(parent=plotterWidgetFound, plotType = 'B0M')
        self.verticalLayout_CV_plotter.addWidget(self.plotterWGT)
        #self.verticalLayout_CV_plotter.addWidget(self.CVplotter.toolbar)
        self.plotter = self.plotterWGT.PlotterCanvas
        self.plotter.preset_B0M()  # just add some labels
     
     
        # B0 slice plotter
        plotterWidgetFound = self.findChild(QtWidgets.QWidget, 'plotter_widget_2d')
        self.plotterWGT2d = Plotter.Plotter(parent=plotterWidgetFound, plotType = 'B0slice')
        self.verticalLayout_2d_plotter.addWidget(self.plotterWGT2d)
        #self.verticalLayout_CV_plotter.addWidget(self.CVplotter.toolbar)
        self.plotter2d = self.plotterWGT2d.PlotterCanvas
        self.plotter.preset_B0slice()  # just add some labels
     
        
        # connect tick box with plotter. on tick plot only one slice
        self.XYcheckBox.stateChanged.connect(self.plot_B0M_slice)
        self.ZXcheckBox.stateChanged.connect(self.plot_B0M_slice)
        self.YZcheckBox.stateChanged.connect(self.plot_B0M_slice)
        self.ShowSphereCheckBox.stateChanged.connect(self.plot_B0M_slice)
        self.ShowMagnetCheckBox.stateChanged.connect(self.plot_B0M_slice)
        self.ShowRingsCheckBox.stateChanged.connect(self.plot_B0M_slice)
        self.PlotSPHCheckBox.stateChanged.connect(self.plot_B0M_slice)
        self.PlotShimFieldCheckBox.stateChanged.connect(self.plot_B0M_slice)
        self.PlotCheapFieldCheckBox.stateChanged.connect(self.plot_B0M_slice)
        self.PlotErrorFieldCheckBox.stateChanged.connect(self.plot_B0M_slice)
        

        
    def load_csv(self):
        # create an empty instance of b0map
        self.b0map = b0.b0()
        # get the file name from the open file dialog
                # open file dialog
        try:
            filename_to_import_csv_data_from, _ = QtWidgets.QFileDialog.getOpenFileName(self, caption="Select B0 data in csv format",
                                                                   directory=self.workingFolder,
                                                                   filter="csv Files (*.csv)")
            self.workingFolder = os.path.split(os.path.abspath(filename_to_import_csv_data_from))[0]

        except:
            print('no filename given, do it again.')
            return 0
        
        # import b0map as an object
        self.b0map = b0.b0()
        self.b0map.import_from_csv(filename_to_import_csv_data_from)
        self.coordinate_transform_btn.clicked.connect(self.change_coords_to_magnet)
        # and print it on the plotter.
        self.plotter.plotPathWithMagnet(self.b0map)
        pass


    def load_b0(self):
        print('open a file dialog, get a b0 scan file name, slice by z, plot first scan')
        # open file dialog
        try:
            self.b0Path, _ = QtWidgets.QFileDialog.getOpenFileName(self, caption="Select B0 data",
                                                                   directory=self.workingFolder,
                                                                   filter="txt Files (*.txt)")
            self.workingFolder = os.path.split(os.path.abspath(self.b0Path))[0]

        except:
            print('no filename given, do it again.')
            return 0

        # open file dialog
        try:
            self.pathPath, _ = QtWidgets.QFileDialog.getOpenFileName(self, caption="Select path data",
                                                                   directory=self.workingFolder,
                                                                   filter="path Files (*.path)")
            self.workingFolder = os.path.split(os.path.abspath(self.pathPath))[0]

        except:
            print('no filename given, do it again.')
            return 0

        # import the b0 as an object
        self.b0map = b0.b0(b0_filename = self.b0Path,path_filename=self.pathPath)
        self.coordinate_transform_btn.clicked.connect(self.change_coords_to_magnet)
        # and print it on the plotter.
        self.plotter.plotPathWithMagnet(self.b0map)
        
        
    def plot_B0M_slice_2d(self):
        # get the ticks
        numticks = 0
        plot_XY_sliceFlag = self.XYcheckBox.isChecked()
        plot_ZX_sliceFlag = self.ZXcheckBox.isChecked()
        plot_YZ_sliceFlag = self.YZcheckBox.isChecked()
        
        plot_RAW_flag = self.PlotRawCheckBox.isChecked() # plot the raw field
        plot_SPH_flag = self.PlotSPHCheckBox.isChecked() # plot the decomposed field
        plot_SHIM_FIELD_flag = self.PlotShimFieldCheckBox.isChecked() # plot the shim field
        plot_ERROR_FIELD_flag = self.PlotErrorFieldCheckBox.isChecked() # plot the error field
        
        
        XY_slice_number = int(self.XYspinBox.value()) if plot_XY_sliceFlag else -1
        ZX_slice_number = int(self.ZXspinBox.value()) if plot_ZX_sliceFlag else -1
        YZ_slice_number = int(self.YZspinBox.value()) if plot_YZ_sliceFlag else -1
        
        if plot_XY_sliceFlag: 
            numticks+=1
            self.plotter2d.set_title('XY plane')
        if plot_ZX_sliceFlag: 
            numticks+=1
            self.plotter2d.set_title('ZX plane')
        if plot_YZ_sliceFlag: 
            numticks+=1
            self.plotter2d.set_title('YZ plane')
            
        if numticks == 0 or numticks > 1:
            print('select only one plane!')
            return 0
        self.plotter2d.plotB0slice_2D(b0map_object=self.b0map, 
                        slice_number_xy=XY_slice_number,
                        slice_number_zx=ZX_slice_number,
                        slice_number_yz=YZ_slice_number, 
                        plot_raw=plot_RAW_flag, 
                        plot_sph=plot_SPH_flag,
                        plot_shim = plot_SHIM_FIELD_flag, 
                        plot_error=plot_ERROR_FIELD_flag)

        

    def plot_B0M_slice(self):
        # get the ticks
        plot_XY_sliceFlag = self.XYcheckBox.isChecked()
        plot_ZX_sliceFlag = self.ZXcheckBox.isChecked()
        plot_YZ_sliceFlag = self.YZcheckBox.isChecked()
        plot_sphere_flag = self.ShowSphereCheckBox.isChecked()
        plot_magnet_flag = self.ShowMagnetCheckBox.isChecked()
        plot_rings_flag = self.ShowRingsCheckBox.isChecked()
        
        plot_RAW_flag = self.PlotRawCheckBox.isChecked() # plot the raw field
        plot_SPH_flag = self.PlotSPHCheckBox.isChecked() # plot the decomposed field
        plot_SHIM_FIELD_flag = self.PlotShimFieldCheckBox.isChecked() # plot the shim field
        plot_CHEAP_FIELD_flag = self.PlotCheapFieldCheckBox.isChecked() # plot the cheap field
        plot_ERROR_FIELD_flag = self.PlotErrorFieldCheckBox.isChecked() # plot the error field


        
        if plot_SPH_flag:
            self.XYspinBox.setMaximum(len(self.b0map.zDim_SPH_fine)-1)               
            self.ZXspinBox.setMaximum(len(self.b0map.yDim_SPH_fine)-1)           
            self.YZspinBox.setMaximum(len(self.b0map.xDim_SPH_fine)-1)       
        if plot_RAW_flag:
            self.XYspinBox.setMaximum(len(self.b0map.xPts)-1)               
            self.ZXspinBox.setMaximum(len(self.b0map.yPts)-1)           
            self.YZspinBox.setMaximum(len(self.b0map.xPts)-1)       
        if plot_SHIM_FIELD_flag or plot_ERROR_FIELD_flag or plot_CHEAP_FIELD_flag:
            self.XYspinBox.setMaximum(len(self.b0map.zDim_SPH_fine)-1)               
            self.ZXspinBox.setMaximum(len(self.b0map.yDim_SPH_fine)-1)           
            self.YZspinBox.setMaximum(len(self.b0map.xDim_SPH_fine)-1)


        
        XY_slice_number = int(self.XYspinBox.value()) if plot_XY_sliceFlag else -1
        ZX_slice_number = int(self.ZXspinBox.value()) if plot_ZX_sliceFlag else -1
        YZ_slice_number = int(self.YZspinBox.value()) if plot_YZ_sliceFlag else -1
        showSphRad = self.b0map.path.radius if plot_sphere_flag else None
        showMagnet = True if plot_magnet_flag else None 
        showRings = True if plot_rings_flag else None
        
        # plot the slices according to the checked boxes
        self.plotter.plotB0Map(b0map_object=self.b0map, 
                               slice_number_xy=XY_slice_number,
                               slice_number_zx=ZX_slice_number,
                               slice_number_yz=YZ_slice_number, 
                               show_sphere_radius=showSphRad,
                               show_magnet = showMagnet,
                               show_rings = showRings, 
                               coordinate_system='magnet',
                               plot_raw = plot_RAW_flag, 
                               plot_sph = plot_SPH_flag,
                               plot_shim = plot_SHIM_FIELD_flag,
                               plot_cheap = plot_CHEAP_FIELD_flag,
                               plot_error = plot_ERROR_FIELD_flag)
        


    def change_coords_to_magnet(self):
        self.b0map.transfer_coordinates_of_the_path_from_cosi_to_magnet()
        self.plotter.plotPathWithMagnet(self.b0map,coordinate_system='magnet')
        
        
        # foolproof checkboxes        
        print(len(self.b0map.zPts))
        
        self.XYspinBox.setMaximum(len(self.b0map.zPts)-1)     
        self.XYspinBox.setValue(round((len(self.b0map.zPts)-1)/2))        
        self.XYspinBox.valueChanged.connect(self.plot_B0M_slice)
        
        print(len(self.b0map.yPts))
        
        self.ZXspinBox.setMaximum(len(self.b0map.yPts)-1)     
        self.ZXspinBox.setValue(round((len(self.b0map.yPts)-1)/2))        
        self.ZXspinBox.valueChanged.connect(self.plot_B0M_slice)

        print(len(self.b0map.yPts))
        
        self.YZspinBox.setMaximum(len(self.b0map.xPts)-1)     
        self.YZspinBox.setValue(round((len(self.b0map.xPts)-1)/2))        
        self.YZspinBox.valueChanged.connect(self.plot_B0M_slice)
        
        # show field inmomogeneity and mean field in the labels
        self.mean_field_label.setText('Init Mean: %.3f [mT]'%float(self.b0map.mean_field))
        self.inhomogeneity_label.setText('Init Homo: %.0f [ppm]'%float(self.b0map.homogeneity))
        
        
        #self.plotter.plotB0Map(self.b0map,slice_number=0,coordinate_system='magnet')
        


    def save_rotated_path_in_a_csv_file(self):
        print('save as file dialog etc, think of the format, Be compatible with the future imports')
        # open file dialog
        try:
            new_csv_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, caption="file name for csv export",
                                                                   directory=self.workingFolder,
                                                                   filter="comma separated Files (*.csv)")
            self.workingFolder = os.path.split(os.path.abspath(new_csv_path))[0]

        except:
            print('no filename given, do it again.')
            return 0
        
        
        # TEMP!
        #self.b0map.make_artificial_field_along_path(coordinates_of_singularity = [25,50,100],radius_of_singularity=50, intensity=47.2, bg = 47)
        self.b0map.make_cylindrical_anomaly_along_x(yz_of_the_cylinder_center=[-50,120],radius_of_cylinder=70,intensity=47.1, bg = 47)
        self.b0map.saveAsCsv_for_comsol(new_csv_path)
        #self.b0map.path.saveAs(new_csv_path)

    def save_shim_magnets_in_rings(self):
        print('file dialog for exporting the ring files for Freecad')
        # open file dialog
        try:
            ring_filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, caption="file name for shim magnets export",
                                                                   directory=self.workingFolder,
                                                                   filter="text files (*.txt)")
            self.workingFolder = os.path.split(os.path.abspath(ring_filename))[0]

        except:
            print('no filename given, do it again.')
            return 0

        try:
            self.b0map.save_rings(ring_filename)
        except Exception as e:
            print(e)


    def load_shim_magnets_in_rings(self):
        print('open filedialog, locate txt file, import ring locations and rotations')
        try:
            filename_to_import_rings_data_from, _ = QtWidgets.QFileDialog.getOpenFileName(self, caption="Select rings data in txt format",
                                                                   directory=self.workingFolder,
                                                                   filter="csv Files (*.txt)")
            self.workingFolder = os.path.split(os.path.abspath(filename_to_import_rings_data_from))[0]

        except:
            print('no filename given, do it again.')
            return 0
        
        # import b0map as an object
        if self.b0map is not None:
            print('updating shim magnets rotations from file')
            self.b0map.update_magnet_rotations(fname = filename_to_import_rings_data_from)





    def export_separately(self):
        print('file dialog for exporting two files for Tom''s script')
        # open file dialog
        try:
            separate_filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, caption="file name for csv export",
                                                                   directory=self.workingFolder,
                                                                   filter="path files for Tom (*.path)")
            self.workingFolder = os.path.split(os.path.abspath(separate_filename))[0]

        except:
            print('no filename given, do it again.')
            return 0

        self.b0map.save_separately(separate_filename)

    # spherical harmonic things
    def fit_sph(self):
        order = int(self.sph_spinbox.value())
        print('FIT SPHERICAL HARMONICS up to %d TO THE DATA,\n DATA MUST BE IN THE MAGNET COORDINATES'%order)
        diameter_of_sphere = self.b0map.path.radius*2
        self.b0map.fitSphericalHarmonics(maxorder=order,dsv=diameter_of_sphere,resol=3)
        resolution_of_sph_fit=int(self.resolution_SPH_spinbox.value())
        self.b0map.interpolateField(resol=resolution_of_sph_fit,dsv=diameter_of_sphere)
        print('spherical harmonic decomposition completed. coefficients extracted.')
        print('now perform field interpolation')
        print('You can now plot decomposed sph by ticking the SPH checkbox.')
        
         # foolproof checkboxes        
        print(len(self.b0map.zPts))
        
        self.XYspinBox.setMaximum(len(self.b0map.zDim_SPH_fine)-1)           
        self.ZXspinBox.setMaximum(len(self.b0map.yDim_SPH_fine)-1)           
        self.YZspinBox.setMaximum(len(self.b0map.xDim_SPH_fine)-1)    
        
    def get_shim_positions(self):
        print('before that, even, plot the shim field.')
        print('First put all magnets in the rings with their b0 along Y.')
        print('and draw magnets in rings with their direcrion vectors.')
        self.b0map.get_shim_positions(dsv_for_opt_percent = int(self.dsv_opt_spinbox.value()),verbose = self.verbose_checkbox.isChecked())
        self.mean_field_shimmed_label.setText('Mean After: %.3f [mT]'%float(self.b0map.mean_field_shimmed))
        self.inhomogeneity_after_label.setText('Homo After: %.0f [ppm]'%float(self.b0map.homogeneity_shimmed))

        