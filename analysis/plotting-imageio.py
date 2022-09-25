from PIL import Image, ImageDraw
import shutil
import os

if os.path.exists("results/temp/anim"):
    shutil.rmtree('results/temp/anim')
    os.makedirs('results/temp/anim')

filename = 'results/2022-09-19/19-17-43'
batch_length = 20
a = 'sep'

# for i in range(10, 5001, 10):
#     print(i)
#     f = f'{filename}/{i}/{a}.png'
#     img = Image.open(f)
#     I1 = ImageDraw.Draw(img)
#     I1.text((28, 36), f"epoch: {i}, batch length = {batch_length}", fill=(0, 0, 0))
#     img.save(f"results/temp/anim/{a}_{i}.png")

#     if i % 500 == 0:
#         batch_length += 10
        

import imageio

# with imageio.get_writer('figures/potential_old.gif', mode='I') as writer:
#     for i in range(0, 501, 5):
#         print(i)
#         filename = f'figures/anim2/{i}.png'
#         image = imageio.imread(filename)
#         writer.append_data(image)


# with imageio.get_writer('figures/traj_old.gif', mode='I') as writer:
#     for i in range(0, 501, 5):
#         print(i)
#         filename = f'figures/anim/{i}.png'
#         image = imageio.imread(filename)
#         writer.append_data(image)

with imageio.get_writer(f'{a}.gif', format='GIF-PIL', mode='I') as writer:
    for i in range(50, 5001, 50):
        print(i)
        # f = f'results/temp/anim/{a}_{i}.png'
        f = f'{filename}/{i}/{a}.png'
        image = imageio.imread(f)
        writer.append_data(image)
