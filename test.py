# from scripts.u_i_a import embed_img as embed_image
# import torch
# emb = embed_image("D:/Bao\Picture/bouncer uia cat.jpg", caption="uia cat")
# uia = emb.image_embeds
# print("Shape: ", uia.shape)

from scripts.u_a_i import embed_text   # new import

# --- Inside main(), near where you compute embeddings ---
prompt = input("Enter your symtoms: ")
if prompt:
    print("Computing text embedding...")
    try:
        text_vector = embed_text(prompt)
        print(f"Generated text embedding of dimension {len(text_vector)}")
    except Exception as e:
        print(f"Text embedding failed: {e}")
        text_vector = None
