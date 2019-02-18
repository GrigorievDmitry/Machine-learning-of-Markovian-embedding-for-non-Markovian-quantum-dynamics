import imageio
#This piece of code generate gif animation from set of pictures
images = []
n = 300 #Number of pictures
filenames = [f'pic/dynamics_fig{i}.png' for i in range(n)]
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('anim.gif', images)
