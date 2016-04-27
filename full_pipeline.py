from nipype.pipeline.engine import Node, Workflow, MapNode
import nipype.interfaces.utility as util
import nipype.interfaces.io as nio
import nipype.interfaces.fsl as fsl
import nipype.interfaces.freesurfer as fs
import nipype.interfaces.nipy as nipy
import nipype.algorithms.rapidart as ra
from nipype.algorithms.misc import TSNR
import nipype.interfaces.ants as ants
from functions import strip_rois_func, get_info, median, motion_regressors, selectindex, fix_hdr
from linear_coreg import create_coreg_pipeline
from nonlinear_coreg import create_nonlinear_pipeline

# read in subjects and file names
subjects=['sub001'] #, 'sub002', 'sub003', 'sub004', 'sub005', 'sub006', 
          # 'sub007', 'sub008', 'sub009', 'sub010', 'sub011', 'sub012', 
          # 'sub013', 'sub014', 'sub015', 'sub016', 'sub017', 'sub018', 
          # 'sub019', 'sub020', 'sub021', 'sub022']
# sessions to loop over
sessions=['session_1' ,'session_2']
# scans to loop over
scans=['rest_full_brain_1', 'rest_full_brain_2']

# directories
working_dir = '/scr/animals1/preproc7t/working_dir/' 
data_dir= '/scr/animals1/preproc7t/data7t/'
out_dir = '/scr/animals1/preproc7t/resting/preprocessed/'
freesurfer_dir = '/scr/animals1/preproc7t/freesurfer/' # freesurfer reconstruction of lowres is assumed

# set fsl output type to nii.gz
fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

# volumes to remove from each timeseries
vol_to_remove = 5

# main workflow
preproc = Workflow(name='func_preproc')
preproc.base_dir = working_dir
preproc.config['execution']['crashdump_dir'] = preproc.base_dir + "/crash_files"

# iterate over subjects
subject_infosource = Node(util.IdentityInterface(fields=['subject']), 
                  name='subject_infosource')
subject_infosource.iterables=[('subject', subjects)]

# iterate over sessions
session_infosource = Node(util.IdentityInterface(fields=['session']), 
                  name='session_infosource')
session_infosource.iterables=[('session', sessions)]

# iterate over scans
scan_infosource = Node(util.IdentityInterface(fields=['scan']), 
                  name='scan_infosource')
scan_infosource.iterables=[('scan', scans)]

# select files
templates={'rest' : 'niftis/{subject}/{session}/{scan}.nii.gz',
           'dicom':'dicoms_example/MR.2.25.130666515827674933471189335089197862909.dcm',
           'uni_lowres' : 'niftis/{subject}/{session}/MP2RAGE_UNI.nii.gz', # changed to lowres
           't1_lowres' : 'niftis/{subject}/{session}/MP2RAGE_T1.nii.gz', # changed to lowres
           'brain_mask' : 'brainmasks/{subject}/ses-1/anat/*.nii.gz',
           }

selectfiles = Node(nio.SelectFiles(templates, base_directory=data_dir),
                   name="selectfiles")

preproc.connect([(subject_infosource, selectfiles, [('subject', 'subject')]),
                 (session_infosource, selectfiles, [('session', 'session')]), 
                 (scan_infosource, selectfiles, [('scan', 'scan')]),  
                 ])

# remove first volumes
remove_vol = Node(util.Function(input_names=['in_file','t_min'],
                                output_names=["out_file"],
                                function=strip_rois_func),
                  name='remove_vol')
remove_vol.inputs.t_min = vol_to_remove

preproc.connect([(selectfiles, remove_vol, [('rest', 'in_file')])])

# get slice time information from example dicom
getinfo = Node(util.Function(input_names=['dicom_file'],
                             output_names=['TR', 'slice_times', 'slice_thickness'],
                             function=get_info),
               name='getinfo')
preproc.connect([(selectfiles, getinfo, [('dicom', 'dicom_file')])])
                 
                 
# simultaneous slice time and motion correction
slicemoco = Node(nipy.SpaceTimeRealigner(),                 
                 name="spacetime_realign")
slicemoco.inputs.slice_info = 2

preproc.connect([(getinfo, slicemoco, [('slice_times', 'slice_times'),
                                       ('TR', 'tr')]),
                 (remove_vol, slicemoco, [('out_file', 'in_file')])])

# compute tsnr and detrend
tsnr = Node(TSNR(regress_poly=2),
               name='tsnr')
preproc.connect([(slicemoco, tsnr, [('out_file', 'in_file')])])
 
# compute median of realigned timeseries for coregistration to anatomy
median = Node(util.Function(input_names=['in_files'],
                       output_names=['median_file'],
                       function=median),
              name='median')
 
preproc.connect([(tsnr, median, [('detrended_file', 'in_files')])])
 
# make FOV mask for later nonlinear coregistration
fov = Node(fsl.maths.MathsCommand(args='-bin',
                                  out_file='fov_mask.nii.gz'),
           name='fov_mask')
preproc.connect([(median, fov, [('median_file', 'in_file')])])

# fix header of brain mask
fixhdr = Node(util.Function(input_names=['data_file', 'header_file'],
                            output_names=['out_file'],
                            function=fix_hdr),
                  name='fixhdr')
preproc.connect([(selectfiles, fixhdr, [('brain_mask', 'data_file'),
                                        ('t1_lowres', 'header_file')]),
                 ])

# biasfield correction of median epi for better registration
biasfield = Node(ants.segmentation.N4BiasFieldCorrection(save_bias=True),
                 name='biasfield')
preproc.connect([(median, biasfield, [('median_file', 'input_image')])])

# perform linear coregistration in ONE step: median2lowres
coreg=create_coreg_pipeline()
coreg.inputs.inputnode.fs_subjects_dir = freesurfer_dir
 
preproc.connect([(selectfiles, coreg, [('uni_lowres', 'inputnode.uni_lowres')]),
                (biasfield, coreg, [('output_image', 'inputnode.epi_median')]),
                (subject_infosource, coreg, [('subject', 'inputnode.fs_subject_id')])
                ])

# perform nonlinear coregistration 
nonreg=create_nonlinear_pipeline()
   
preproc.connect([(selectfiles, nonreg, [('t1_lowres', 'inputnode.t1_lowres')]),
                 (fixhdr, nonreg, [('out_file', 'inputnode.brain_mask')]),
                 (fov, nonreg, [('out_file', 'inputnode.fov_mask')]),
                 (coreg, nonreg, [('outputnode.epi2lowres_lin', 'inputnode.epi2lowres_lin'),
                                  ('outputnode.epi2lowres_lin_itk', 'inputnode.epi2lowres_lin_itk')])
                  ])

# make wm/csf mask from segmentations and erode, and medial wall mask
fs_import = Node(interface=nio.FreeSurferSource(),
                     name = 'fs_import')

fs_import.inputs.subjects_dir = freesurfer_dir
preproc.connect([(subject_infosource, fs_import, [('subject', 'subject_id')])])

wmmask = Node(fs.Binarize(wm_ven_csf = True,
                          erode = 2,
                          out_type = 'nii.gz',
                          binary_file='wm_mask.nii.gz'), 
               name='wmmask')


preproc.connect([(fs_import, wmmask, [('aseg', 'in_file')])                 
                 ])

# merge struct2func transforms into list
translist_inv = Node(util.Merge(2),name='translist_inv')
preproc.connect([(coreg, translist_inv, [('outputnode.epi2lowres_lin_itk', 'in1')]),
                 (nonreg, translist_inv, [('outputnode.epi2lowres_invwarp', 'in2')])])
   
# merge images into list
structlist = Node(util.Merge(2),name='structlist')
preproc.connect([(fixhdr, structlist, [('out_file', 'in1')]),
                 (wmmask, structlist, [('binary_file', 'in2')])                 
                 ])
   
# project brain mask, wm/csf masks, t1 and subcortical mask in functional space
struct2func = MapNode(ants.ApplyTransforms(dimension=3,
                                         invert_transform_flags=[True, False],
                                         interpolation = 'NearestNeighbor'),
                    iterfield=['input_image'],
                    name='struct2func')

   
preproc.connect([(structlist, struct2func, [('out', 'input_image')]),
                 (translist_inv, struct2func, [('out', 'transforms')]),
                 (median, struct2func, [('median_file', 'reference_image')]),
                 ])


# perform artefact detection
artefact=Node(ra.ArtifactDetect(save_plot=True,
                                use_norm=True,
                                parameter_source='NiPy',
                                mask_type='file',
                                norm_threshold=1,
                                zintensity_threshold=3,
                                use_differences=[True,False]
                                ),
             name='artefact')
   
preproc.connect([(slicemoco, artefact, [('out_file', 'realigned_files'),
                                        ('par_file', 'realignment_parameters')]),
                 (struct2func, artefact, [(('output_image', selectindex, [0]), 'mask_file')]),
                 ])
  
# calculate motion regressors
motreg = MapNode(util.Function(input_names=['motion_params', 'order','derivatives'],
                            output_names=['out_files'],
                            function=motion_regressors),
                 iterfield=['order'],
                 name='motion_regressors')
motreg.inputs.order=[1] #,2
motreg.inputs.derivatives=1
preproc.connect([(slicemoco, motreg, [('par_file','motion_params')])])
  
# sink relevant files
sink = Node(nio.DataSink(parameterization=True),
             name='sink')

sink.inputs.base_directory = out_dir

preproc.connect([#(scan_infosource, sink, [('scan', 'container')]),
                 #(session_infosource, sink, [('session', 'container')]),
                 #(subject_infosource, sink, [(('subject', makebase, out_dir), 'base_directory')]),
                 (remove_vol, sink, [('out_file', 'realignment.@raw_file')]),
                 (slicemoco, sink, [('out_file', 'realignment.@realigned_file'),
                                    ('par_file', 'confounds.@orig_motion')]),
                 (tsnr, sink, [('tsnr_file', 'realignment.@tsnr')]),
                 (median, sink, [('median_file', 'realignment.@median')]),
                 (biasfield, sink, [('output_image', 'realignment.@biasfield')]),
                 (coreg, sink, [('outputnode.uni_lowres', 'registration.@uni_lowres'),
                                ('outputnode.epi2lowres_lin_mat','registration.@epi2lowres_lin_mat'),
                                ('outputnode.epi2lowres_lin_dat','registration.@epi2lowres_lin_dat'),
                                ('outputnode.epi2lowres_lin', 'registration.@epi2lowres_lin'),
                                ('outputnode.epi2lowres_lin_itk', 'registration.@epi2lowres_lin_itk'),
                               ]),
                (nonreg, sink, [('outputnode.epi2lowres_warp', 'registration.@epi2lowres_warp'),
                                ('outputnode.epi2lowres_invwarp', 'registration.@epi2lowres_invwarp'),
                                ('outputnode.epi2lowres_nonlin', 'registration.@epi2lowres_nonlin')]),
                (struct2func, sink, [(('output_image', selectindex, [0,1]), 'mask.@masks')]),
                (artefact, sink, [('norm_files', 'confounds.@norm_motion'),
                                  ('outlier_files', 'confounds.@outlier_files'),
                                  ('intensity_files', 'confounds.@intensity_files'),
                                  ('statistic_files', 'confounds.@outlier_stats'),
                                  ('plot_files', 'confounds.@outlier_plots')]),
                 (motreg, sink, [('out_files', 'confounds.@motreg')])
                 ])
    
preproc.run(plugin='MultiProc', plugin_args={'n_procs' : 1})
