import os, yaml, time
from glob import glob

#load configs from yaml file
def get_config(config):
    with open(config,'r') as stream:
        return yaml.load(stream)

# save configuration file
def write_config(config, outfile):
    with open(outfile,'w') as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False)

def check_folder(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir

def mkdir_output_train(args):
    dir_out_base = args['dir_out']
    dir_log = args['output']['dir_log']
    dir_config = args['output']['dir_config']
    dir_sample = args['output']['dir_sample']
    dir_checkpoint = args['output']['dir_checkpoint']

    dir_log = os.path.join(dir_out_base, dir_log)
    dir_config = os.path.join(dir_out_base, dir_config)
    dir_sample = os.path.join(dir_out_base, dir_sample)
    dir_checkpoint = os.path.join(dir_out_base, dir_checkpoint)

    check_folder(dir_log)
    check_folder(dir_config)
    check_folder(dir_sample)
    check_folder(dir_checkpoint)

    write_config(args, os.path.join(dir_config, 'configs.yaml'))
    return dir_log, dir_config, dir_sample, dir_checkpoint

def mkdir_output_test(args):
    dir_out_base = args['dir_out']
    dir_result = args['output']['dir_result']
    dir_result = os.path.join(dir_out_base, dir_result)
    check_folder(dir_result)
    return dir_result

def get_file_list(path_file):
    list_file = []
    if os.path.isdir(path_file):
        list_file = glob(path_file + '/*.*')
    else:
        list_file.append(path_file)
    return list_file