import taichi as ti
import argparse
import os
import imageio
from mass_spring_model import Mass_spring

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="implicit")
args = parser.parse_args()

ti.init(arch=ti.cpu)

vertices = ti.Vector.field(3, dtype=ti.f32, shape=2) 
indices = ti.field(int, shape=2)

mass_spring_simple = Mass_spring(2, 1, 0.005, 40, 10)

@ti.kernel
def update_vertices():
    for i in vertices:
        vertices[i] = mass_spring_simple.pos[i]

mass_spring_simple.e2v[0] = [0, 1]
mass_spring_simple.l_i[0] = 1.0
mass_spring_simple.pos[0] = [0, 0, 0]
mass_spring_simple.pos[1] = [0, 2, 0]
mass_spring_simple.vel[0] = [-1, 0, 0]
mass_spring_simple.vel[1] = [1, 0, 0]
indices[0] = 0
indices[1] = 1

substeps = 10
def main():
    window = ti.ui.Window("Taichi Simulation on GGUI", (768, 768),
                          vsync=True)
    canvas = window.get_canvas()
    canvas.set_background_color((1, 1, 1))
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()

    update_vertices()

    tot_step = 0
    save_path = f"../imgs/mass_spring_simple_{args.mode}"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    frames = []
    while window.running:
        if tot_step > 200:
            break
        tot_step += 1

        for i in range(substeps):
            mass_spring_simple.step(args.mode)

        update_vertices()

        camera.position(0, -2.5, 3)
        camera.lookat(0, 1.5, 0)
        scene.set_camera(camera)

        scene.point_light(pos=(0.5, 0.5, 3), color=(1, 1, 1))
        scene.ambient_light((0.5, 0.5, 0.5))

        scene.lines(vertices, 0.05, indices, color=(0, 0, 1))
        scene.particles(vertices, 0.1, color=(1, 0, 0))
        canvas.scene(scene)

        window.save_image(os.path.join(save_path, f"{tot_step}.png"))
        window.show()
    
    for i in range(1, tot_step+1):
        filename = os.path.join(save_path, f"{i}.png")
        frames.append(imageio.imread(filename))
    
    gif_name = os.path.join(save_path, f"GIF.gif")
    imageio.mimsave(gif_name, frames, 'GIF', duration=0.2)
    

main()
