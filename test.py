from scripts.u_i_a import embed_img as embed_image
import torch
emb = embed_image("D:/Bao\Picture/bouncer uia cat.jpg", caption="uia cat")
uia = emb.image_embeds
print("Shape: ", uia.shape)