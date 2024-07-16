from typing import ClassVar
from crewai_tools import BaseTool
from diffusers import DiffusionPipeline


class IconGeneratorTool(BaseTool):
    name: str = "Devoteam Icon Generator"
    description: str = (
        "Accepts prompt as argument and use this prompt to generate relevant icon image."
    )
    pipeline: ClassVar[DiffusionPipeline] = DiffusionPipeline.from_pretrained("hardik-id/devoteam-icon-generator")
    # pipeline.to("cuda")

    def _run(self, prompt: str) -> str:
        image = self.pipeline("An image of a squirrel in Picasso style").images[0];
        image.save("squirrel_picasso.png")
        print(f"Image generated and saved as squirrel_picasso.png")
        image.show()
        return image
