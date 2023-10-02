import yaml
import sys

def modify_yaml(n_approx):
    with open("naslib/runners/predictors/gp_config.yaml", 'r') as stream:
        data_loaded = yaml.safe_load(stream)

    data_loaded['gp_heat']['n_approx'] = n_approx

    with open("naslib/runners/predictors/gp_config.yaml", 'w') as stream:
        yaml.dump(data_loaded, stream)

if __name__ == "__main__":
    n_approx = int(sys.argv[1])
    modify_yaml(n_approx)
