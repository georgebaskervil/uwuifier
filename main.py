from animesd import generate_anime_image
from squareimage import crop_image_to_square
from random import randint
from animesharp import upscale_anime_image

input_image_path = "/Users/george/uwuifier/images/me.JPG"
squared_image_path = "/Users/george/uwuifier/images/me_square.JPG"
crop_image_to_square(input_image_path, squared_image_path)

anime_prompt = """
    anime style, futuristic tactical gear, dynamic lighting, intricate detail, background,
    detailed clothing, expressive accessories, high contrast effects, sleek modern design,
    luminous highlights, soft shadows, precise textures, rhythmic patterns, fluid composition
    """
anime_negative_prompt = """
    low quality, blurry, deformed, duplicate, out of frame, unnatural lighting,
    watermark, text, error, grainy textures, oversaturated colors, unnatural proportions, poorly lit,
    particularly detailed background, unnatural facial composition, inconsistent skin texture, unnatural face shapes.
    """
random_seed = randint(0, 1000000)
anime_output_dir = "/Users/george/uwuifier/images"
anime_image_path = "/Users/george/uwuifier/images/me_square_anime.png"

generate_anime_image(
    squared_image_path,
    output_dir=anime_output_dir,
    output_image_path=anime_image_path,
    prompt=anime_prompt,
    seed=random_seed,
    negative_prompt=anime_negative_prompt,
)

# Add upscaling step
upscaled_anime_path = "/Users/george/uwuifier/images/me_square_anime_upscaled.png"
upscale_anime_image(anime_image_path, upscaled_anime_path)
