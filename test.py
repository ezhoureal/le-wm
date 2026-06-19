import hydra
from omegaconf import DictConfig
import stable_worldmodel as swm


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    
    model = swm.wm.utils.load_pretrained(cfg.policy)
    solver = hydra.utils.instantiate(cfg.solver, model=model)

if __name__ == "__main__":
    run()