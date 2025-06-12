from flyvis.datasets.FT3D import RenderedFlyingThings3D

root = "/mnt/s/datasets/FlyingThings3D"
RenderedFlyingThings3D(tasks=["flow", "rgb"], ft3d_path=root)
