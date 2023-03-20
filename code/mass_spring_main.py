import taichi as ti
import argparse
import os
import imageio
from mass_spring_model import Mass_spring

parser = argparse.ArgumentParser()
parser.add_argument("--exp_id", type=int, default=0)
parser.add_argument("--mode", type=str, default="implicit")
parser.add_argument("--N", type=int, default=31)
parser.add_argument("--Kl", type=float, default=1000.0)
parser.add_argument("--dt", type=float, default=5e-3)
parser.add_argument("--mass", type=float, default=10.0)
parser.add_argument("--save", action="store_true", default = False)
args = parser.parse_args()

ti.init(arch=ti.cpu)

mass = 10.0
dt = args.dt
if args.mode == "implicit":
    dt = 5e-2
Kl = 1000.0
N = args.N
dx = 3.0 / N
NV_complex = (N + 1)**2
NE_complex = (N + 1) * N * 2 + N**2 * 2
vertices = ti.Vector.field(3, dtype=ti.f32, shape=NV_complex) 
indices = ti.field(int, shape=NE_complex * 2)

mass_spring = Mass_spring(NV_complex, NE_complex, dt, Kl, mass)

@ti.kernel
def update_vertices():
    for i in vertices:
        vertices[i] = mass_spring.pos[i]

@ti.kernel
def init_complex():
    for i, j in ti.ndrange(N + 1, N + 1):
        k = i * (N + 1) + j        
        mass_spring.pos[k] = ti.Vector([i * dx, j * dx, ti.random(float)*0.001])
        mass_spring.vel[k] = ti.Vector([0, 0, 0])

    for i, j in ti.ndrange(N + 1, N):
        k = i * N + j
        a = i * (N + 1) + j
        b = i * (N + 1) + j + 1
        indices[k * 2] = a
        indices[k * 2 + 1] = b
        mass_spring.e2v[k] = [a, b]
        mass_spring.l_i[k] = dx

    for i, j in ti.ndrange(N, N + 1):
        k = i * (N + 1) + j + (N + 1) * N
        a = i * (N + 1) + j
        b = (i + 1) * (N + 1) + j
        indices[k * 2] = a
        indices[k * 2 + 1] = b
        mass_spring.e2v[k] = [a, b]
        mass_spring.l_i[k] = dx

    for i, j in ti.ndrange(N, N):
        k = (i * N + j) * 2 + (N + 1) * N * 2
        a = i * (N + 1) + j
        b = i * (N + 1) + j + 1
        c = (i + 1) * (N + 1) + j
        d = (i + 1) * (N + 1) + j + 1
        indices[k * 2] = a
        indices[k * 2 + 1] = d
        indices[(k + 1) * 2] = b
        indices[(k + 1) * 2 + 1] = c
        mass_spring.e2v[k] = [a, d]
        mass_spring.e2v[k + 1] = [b, c]
        mass_spring.l_i[k] = dx * ti.math.sqrt(2.0)
        mass_spring.l_i[k + 1] = dx * ti.math.sqrt(2.0)

substeps = 100   
if args.mode == "implicit":
    substeps = 1
def main():
    window = ti.ui.Window("Taichi Simulation on GGUI", (768, 768),
                          vsync=True)
    canvas = window.get_canvas()
    canvas.set_background_color((1, 1, 1))
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()

    init_complex()
    update_vertices()

    # for i in range(10):
    #     print(f2v[i])
    mass_spring.manipulate_acc[None] = ti.Vector([0, 0, 100])
    tot_step = 0
    save_path = f"../imgs/mass_spring_complex_{args.mode}_{args.exp_id}"
    if args.save:
        if not os.path.exists(save_path):
            os.mkdir(save_path)
    frames = []
    while window.running:
        if tot_step > 300:
            break
        tot_step += 1

        for i in range(substeps):
            mass_spring.step(args.mode)

        update_vertices()

        camera.position(1.5, -2.5, 3)
        camera.lookat(1.5, 1.5, 0)
        scene.set_camera(camera)

        scene.point_light(pos=(0.5, 0.5, 3), color=(1, 1, 1))
        scene.ambient_light((0.5, 0.5, 0.5))

        scene.lines(vertices, 0.005, indices, color=(0, 0, 1))
        scene.particles(vertices, 0.01, color=(1, 0, 0))
        canvas.scene(scene)
        if args.save:
            window.save_image(os.path.join(save_path, f"{tot_step}.png"))
        window.show()
    
    if args.save:
        for i in range(1, tot_step+1):
            filename = os.path.join(save_path, f"{i}.png")
            frames.append(imageio.imread(filename))
        
        gif_name = os.path.join(save_path, f"GIF.gif")
        imageio.mimsave(gif_name, frames, 'GIF', duration=0.2)
    

main()
