import wandb

class RunConfig:
    def __init__(self, args: dict):
        self.entity = args.wandb.entity
        self.project = args.wandb.project
        self.run_id = args.wandb.run_id
        self.model_id = None
        self.vbll = None
        self._get_config()

    def _get_config(self) -> None:
        api = wandb.Api()
        try:
            run = api.run(f"{self.entity}/{self.project}/{self.run_id}")
            self.model_id = run.config['model_name']
            self.vbll = run.config['knowledge_distillation']
        except Exception as e:
            raise ValueError(f"Failed to retrieve run configuration: {e}")