
import os
import sys
import math
import pandas as pd
import torch
import torch.nn.functional as F
from ray.train.lightning import RayTrainReportCallback

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader


import lightning as L
from lightning import Trainer
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

import mlflow
import mlflow.pytorch

import ray
import ray.train.lightning
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig, FailureConfig

MLFLOW_URI = "http://129.114.26.75:8000"
EXPERIMENT_NAME = "Rodent_GAT"

DEFAULT_CONFIG = {
    "window_size":    3,
    "hidden_channels": 16,
    "out_channels":    1,
    "heads":           4,
    "lr":              1e-2,
    "batch_size":      2,
    "num_epochs":     20,
    "boro":           "Brooklyn",
}

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0 
    φ1, φ2 = map(math.radians, (lat1, lat2))
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)
    a = math.sin(Δφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(Δλ/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def get_edge_index(latitudes, longitudes, cutoff_km=1.0):
    coords = list(zip(latitudes, longitudes))
    edges = []
    N = len(coords)
    for i in range(N):
        for j in range(i+1, N):
            if haversine(*coords[i], *coords[j]) <= cutoff_km:
                edges += [[i, j], [j, i]]
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()

def prepare_data(window_size, boro):
    pieces = []
    for chunk in pd.read_csv(
        "/mnt/rodent/data/all_radius.csv",
        chunksize=500_000
    ):
        pieces.append(chunk[chunk.boro == boro])
    df = pd.concat(pieces, ignore_index=True)

    df = df[[
        "camis", "dba", "latitude", "longitude", "month", "rat_complaints_0.5mi"
    ]]
    df["month"] = pd.to_datetime(df["month"]).dt.to_period("M")
    df = (
        df[df.month > "2000-01"]
        .drop_duplicates(subset=["camis", "dba", "month"])
        .sort_values(["camis", "month"])
    )

    camis_list = sorted(df["camis"].unique())
    months     = sorted(df["month"].unique())

    idx = pd.MultiIndex.from_product([camis_list, months], names=["camis", "month"])
    df_full = df.set_index(["camis","month"]).reindex(idx, fill_value=0).reset_index()
    data_dict = {
        (r["camis"], r["month"]): r["rat_complaints_0.5mi"]
        for _, r in df_full.iterrows()
    }

    train_months = months[:-9]
    test_months  = months[-9:]

    def make_samples(month_list):
        samples = []
        for t in range(window_size, len(month_list)):
            xs, ys, zcs = [], [], []
            for z in camis_list:
                feats = [data_dict[(z, month_list[p])]
                         for p in range(t-window_size, t)]
                label = data_dict[(z, month_list[t])]
                xs.append(feats); ys.append(label); zcs.append(z)
            samples.append({
                "x":     torch.tensor(xs, dtype=torch.float),
                "y":     torch.tensor(ys, dtype=torch.float),
                "camis": zcs,
                "month": month_list[t],
            })
        return samples

    return (
        make_samples(train_months),
        make_samples(test_months),
        camis_list,
        {
            r["camis"]: {"latitude":r["latitude"], "longitude":r["longitude"]}
            for _, r in df[["camis","latitude","longitude"]]
                        .drop_duplicates("camis").iterrows()
        }
    )

class LightningGAT(L.LightningModule):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, lr):
        super().__init__()
        from torch_geometric.nn import GATConv
        self.save_hyperparameters()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.conv2 = GATConv(hidden_channels*heads, out_channels, heads=1, concat=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.softplus(x).squeeze()

    def _step(self, batch, stage):
        loss = F.mse_loss(self(batch), batch.y)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")
    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")
    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def train_func(config):
    boro = config['boro']
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlf_logger = MLFlowLogger(
        experiment_name=EXPERIMENT_NAME, tracking_uri=MLFLOW_URI, log_model=False
    )
    mlf_logger.log_hyperparams(config)

    # train_samples = torch.load(f"/mnt/rodent/akshay/train_samples_"+boro+".pt", weights_only=False)
    # test_samples = torch.load(f"/mnt/rodent/akshay/test_samples_"+boro+".pt", weights_only=False)
    # import pickle

    # with open(f"/mnt/rodent/akshay/camis_list_"+boro+".pkl", 'rb') as f:
    #     camis_list = pickle.load(f)
   
    # import json
    # with open(f"/mnt/rodent/akshay/coord_map_"+boro +".json", 'r') as f:
    #     coord_map = json.load(f)

    train_samples, test_samples, camis_list, coord_map = prepare_data(
        config["window_size"], config["boro"]
    )
    print("Read done")
    lats = [coord_map[str(c)]["latitude"]  for c in camis_list]
    lons = [coord_map[str(c)]["longitude"] for c in camis_list]
    ei   = get_edge_index(lats, lons)

    train_loader = GeoDataLoader(
        [Data(x=s["x"], edge_index=ei, y=s["y"]) for s in train_samples],
        batch_size=config["batch_size"], shuffle=True
    )
    val_loader = GeoDataLoader(
        [Data(x=s["x"], edge_index=ei, y=s["y"]) for s in test_samples],
        batch_size=config["batch_size"]
    )

    model = LightningGAT(
        in_channels=config["window_size"] + 12,
        hidden_channels=config["hidden_channels"],
        out_channels=config["out_channels"],
        heads=config["heads"],
        lr=config["lr"],
    )
    ckpt_cb = ModelCheckpoint(
        monitor="val_loss", mode="min", save_top_k=1, filename="best"
    )
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, mode="min"),
        ckpt_cb,
        ray.train.lightning.RayTrainReportCallback(),
    ]

    trainer = Trainer(
        max_epochs=config["num_epochs"],
        devices="auto", accelerator="auto",
        precision="16-mixed",
        strategy=ray.train.lightning.RayDDPStrategy(),
        plugins=[ray.train.lightning.RayLightningEnvironment()],
        logger=mlf_logger,
        callbacks=callbacks,
    )
    trainer = ray.train.lightning.prepare_trainer(trainer)

    ckpt = ray.train.get_checkpoint()
    if ckpt:
        with ckpt.as_directory() as d:
            trainer.fit(model, train_loader, val_loader, ckpt_path=os.path.join(d, "checkpoint.ckpt"))
    else:
        trainer.fit(model, train_loader, val_loader)

    trainer.test(model, val_loader)

    best_path = ckpt_cb.best_model_path
    if best_path:
        best = LightningGAT.load_from_checkpoint(best_path).eval()
        rows = []
        for s in test_samples:
            data = Data(x=s["x"], edge_index=ei)
            with torch.no_grad():
                preds = best(data).numpy()
            for cam, act, pr in zip(s["camis"], s["y"].numpy(), preds):
                rows.append({"camis":cam, "month":str(s["month"]), "actual":float(act), "pred":float(pr)})
        pd.DataFrame(rows).to_csv("predictions_by_month.csv", index=False)
        print("▶️ predictions_by_month.csv saved")
        with mlflow.start_run(run_id=run_id):
            mlflow.pytorch.log_model(
                pytorch_model=best,
                artifact_path="best_model",           
                registered_model_name=f"Graph_" +boro    
            )
    else:
        print("⚠️ No checkpoint to export")

    run_id = mlf_logger.run_id
    

if __name__ == "__main__":
    boro = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG["boro"]
    DEFAULT_CONFIG["boro"] = boro

    run_cfg = RunConfig(
        name=f"GAT_{boro}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
        storage_path="s3://ray",
        failure_config=FailureConfig(max_failures=2),
    )
    scale_cfg = ScalingConfig(num_workers=2, use_gpu=True, resources_per_worker={"GPU":1, "CPU":120})

    trainer = TorchTrainer(
        train_func,
        train_loop_config=DEFAULT_CONFIG,
        scaling_config=scale_cfg,
        run_config=run_cfg
    )
    result = trainer.fit()
    print("Metrics:", result.metrics)
