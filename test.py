import hydra
from omegaconf import DictConfig
import stable_worldmodel as swm
import stable_pretraining as spt


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):

    array = ['{"', 'output_contract', '": {"desired', '_contact_side', '": "side', '", "format', '": "Return', ' exactly one JSON', ' object and no', ' other text.",', ' "next_sub', 'goal": {"block', '_pose_delta', '": {"d', 'theta": ', '0.05', ', "dx', '": -12', '.0, "', 'dy":', ' 4.', '0},', ' "pusher_xy', '": [0.', '600,', ' 0.', '150]},', ' "object_relation', '": "edge', ' of the block', '", "phase', '": "appro', 'ach", "ration', 'ale": "', 'Approach block', ' to position', ' for lateral', ' push towards', ' goal', '."', '}}']
    


if __name__ == "__main__":
    run()
