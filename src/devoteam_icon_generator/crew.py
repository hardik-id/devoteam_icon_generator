from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools.tools.directory_read_tool import directory_read_tool
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
import os
from crewai_tools import DirectoryReadTool, FileReadTool

from devoteam_icon_generator.tools.icon_generator_tool import IconGeneratorTool

# Uncomment the following line to use an example of a custom tool
# from devoteam_icon_generator.tools.custom_tool import MyCustomTool

# Check our tools documentations for more information on how to use them
# from crewai_tools import SerperDevTool

load_dotenv()

@CrewBase
class DevoteamIconGeneratorCrew():
	"""DevoteamIconGenerator crew"""
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'
	azure_llm = AzureChatOpenAI(
		azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
		api_key=os.environ.get("AZURE_OPENAI_KEY"),
		api_version=os.environ.get("AZURE_OPENAI_VERSION"),
		azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
	)
	directory_read_tool = DirectoryReadTool(directory='./icons')
	file_read_tool = FileReadTool()
	@agent
	def designer(self) -> Agent:
		return Agent(
			config=self.agents_config['devoteam_icon_designer'],
			# tools=[MyCustomTool()], # Example of custom tool, loaded on the beginning of file
			llm=self.azure_llm,
			allow_delegation=False,
			verbose=True
		)

	@agent
	def generator(self) -> Agent:
		return Agent(
			config=self.agents_config['devoteam_icon_generator'],
			llm=self.azure_llm,
			tools=[IconGeneratorTool()],
			allow_delegation=False,
			verbose=True
		)
	@agent
	def evaluator(self) -> Agent:
		return Agent(
			config=self.agents_config['devoteam_icon_evaluator'],
			llm=self.azure_llm,
			tools=[self.directory_read_tool, self.file_read_tool],
			allow_delegation=False,
			verbose=True
		)

	@task
	def design_task(self) -> Task:
		return Task(
			config=self.tasks_config['design_icon_task'],
			agent=self.designer()
		)

	@task
	def generate_task(self) -> Task:
		return Task(
			config=self.tasks_config['generate_icon_task'],
			agent=self.generator(),
			output_file='report.md'
		)
	@task
	def evaluate_task(self) -> Task:
		return Task(
			config=self.tasks_config['evaluate_icon_task'],
			agent=self.evaluator(),
			output_file='report.md'
		)
	@crew
	def crew(self) -> Crew:
		"""Creates the DevoteamIconGenerator crew"""
		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=2,
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)