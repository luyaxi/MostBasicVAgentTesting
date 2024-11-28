import fire
import json

from mbvat.visualization import draw_color_coverage,draw_res_correctness

def main(
    infile:str,
    upscale:int = 1,
    save: bool = True,
):
    with open(infile) as f:
        data = json.load(f)
    if save:
        draw_res_correctness(data,save_path=infile.replace(".json","_loc.png"),upscale=upscale)
    else:
        draw_res_correctness(data,upscale=upscale)
    
    if data["meta"]["Colorful"]:
        if save:
            draw_color_coverage(data,save_path=infile.replace(".json","_color.png"))
        else:
            draw_color_coverage(data)

    
if __name__ == "__main__":
    fire.Fire(main)