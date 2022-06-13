
import wandb


if __name__ == '__main__':

    # run = wandb.Api().run("{entity}/{project}/{run-id}"})
    runs = wandb.Api().runs("lauris_bell/bell")
    for run in runs:
        # run.tags.append("100px")  # you can choose tags based on run data here
        run.tags.append("olddb")
        run.update()