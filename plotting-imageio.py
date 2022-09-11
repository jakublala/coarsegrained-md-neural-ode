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

with imageio.get_writer('angvel.gif', mode='I') as writer:
    for i in range(10, 601, 10):
        print(i)
        filename = f'results/2022-09-11/11-38-49/{i}/angvel1.png'
        image = imageio.imread(filename)
        writer.append_data(image)
