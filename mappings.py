import pandas as pd
from nipype.pipeline.engine import Node, Workflow, MapNode
import nipype.interfaces.utility as util
import nipype.interfaces.io as nio
import nipype.interfaces.fsl as fsl
import nipype.interfaces.freesurfer as fs
import nipype.interfaces.afni as afni
import nipype.interfaces.nipy as nipy
import nipype.algorithms.rapidart as ra
from nipype.algorithms.misc import TSNR
import nipype.interfaces.ants as ants
import nilearn.image as nli
from functions import strip_rois_func, get_info, median, motion_regressors, extract_noise_components, selectindex, fix_hdr
from linear_coreg import create_coreg_pipeline
from nonlinear_coreg import create_nonlinear_pipeline


# read in subjects and file names
df=pd.read_csv('/scr/ilz3/myelinconnect/subjects.csv', header=0)
subjects_db=list(df['DB'])


# sessions to loop over
sessions=['rest1_1'] # ,'rest1_2', 'rest2_1', 'rest2_2']

# directories
working_dir = '/scr/ilz3/myelinconnect/working_dir/' 
data_dir= '/scr/ilz3/myelinconnect/'
out_dir = '/scr/ilz3/myelinconnect/mappings/'

# set fsl output type to nii.gz
fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

# volumes to remove from each timeseries
vol_to_remove = 5

# bandpass filter cutoffs in HZ, and TR to calculate sigma
#TR=3.0
#highpass=0.01
#lowpass=0.1

# main workflow
mappings = Workflow(name='mappings')
mappings.base_dir = working_dir
mappings.config['execution']['crashdump_dir'] = mappings.base_dir + "/crash_files"

# iterate over subjects
subject_infosource = Node(util.IdentityInterface(fields=['subject']), 
                  name='subject_infosource')
subject_infosource.iterables=[('subject', subjects_db)]

# iterate over sessions
session_infosource = Node(util.IdentityInterface(fields=['session']), 
                  name='session_infosource')
session_infosource.iterables=[('session', sessions)]

# select files
templates={'median': 'resting/preprocessed/{subject}/{session}/realignment/corr_{subject}_{session}_roi_detrended_median_corrected.nii.gz',
           't1_mapping': 'mappings/t1/{subject}*T1_Images_merged_mapping.nii.gz',
           't1_highres' : 'struct/t1/{subject}*T1_Images_merged.nii.gz',
           'median_mapping' : 'mappings/{session}/corr_{subject}_{session}_*mapping.nii.gz',
           'epi2highres_lin_itk' : 'resting/preprocessed/{subject}/{session}/registration/epi2highres_lin.txt',
           'epi2highres_warp':'resting/preprocessed/{subject}/{session}/registration/transform0Warp.nii.gz',
           'epi2highres_invwarp':'resting/preprocessed/{subject}/{session}/registration/transform0InverseWarp.nii.gz',
           
           }    
selectfiles = Node(nio.SelectFiles(templates, base_directory=data_dir),
                   name="selectfiles")

mappings.connect([(subject_infosource, selectfiles, [('subject', 'subject')]),
                 (session_infosource, selectfiles, [('session', 'session')])
                 ])

# merge func2struct transforms into list
translist_forw = Node(util.Merge(2),name='translist_forw')
mappings.connect([(selectfiles, translist_forw, [('epi2highres_lin_itk', 'in2')]),
                 (selectfiles, translist_forw, [('epi2highres_warp', 'in1')])])
   
# merge struct2func transforms into list
translist_inv = Node(util.Merge(2),name='translist_inv')
mappings.connect([(selectfiles, translist_inv, [('epi2highres_lin_itk', 'in1')]),
                 (selectfiles, translist_inv, [('epi2highres_invwarp', 'in2')])])


# project functional mapping to anatomical space
func2struct = Node(ants.ApplyTransforms(invert_transform_flags=[False, False],
                                        dimension=3,
                                        input_image_type=3,
                                        interpolation='Linear'),
                    name='func2struct')
    
mappings.connect([(selectfiles, func2struct, [('median_mapping', 'input_image'),
                                                ('t1_highres', 'reference_image')]),
                    (translist_forw, func2struct, [('out', 'transforms')]),
                 ])


# project structural mapping to functional space
struct2func = Node(ants.ApplyTransforms(invert_transform_flags=[True, False],
                                        dimension=3,
                                        input_image_type=3,
                                        interpolation='Linear'),
                    name='struct2func')
    
mappings.connect([(selectfiles, struct2func, [('t1_mapping', 'input_image'),
                                                ('median', 'reference_image')]),
                    (translist_inv, struct2func, [('out', 'transforms')]),
                 ])

  
# sink relevant files
sink = Node(nio.DataSink(parameterization=False,
                               base_directory=out_dir),
             name='sink')

mappings.connect([(func2struct, sink, [('output_image', 'rest1_1_to_t1.@func')]),
                    (struct2func, sink, [('output_image', 't1_to_rest1_1.@anat')])
                   ])
    
mappings.run(plugin='MultiProc', plugin_args={'n_procs' : 9})
