from pydantic import BaseModel, Field, UUID4, InstanceOf, computed_field, Json
import uuid
from squadAI.tools import Tool
from squadAI.reactAgent import ReactAgent


class Agent(BaseModel):
    """
    This is the base class for creating actionable agent from react agent

    Attribute:
    ----------
    id: unique id for each agent created using this class
    backstory: context about the purpose of the agent, tasks it needs to complete etc.
    tools: list of tools that agent will require to perform the task

    Methods:
    react_agent: returns the instance of an react agent
    create_user_prompt: based on the user query, it creates the user query and returns it. User prompt contains the task, expected output and the relevant context
    run: it triggers the invoke method within react agent to perform the task
    """

    id: UUID4 = Field(
        default_factory=uuid.uuid4, description="Provides the unique id for "
    )
    backstory: str = Field(default=None, description="Provide character to the agent")
    tools: list[Tool] = []

    @computed_field(return_type=InstanceOf[ReactAgent])
    @property
    def react_agent(self):
        return ReactAgent(tools=self.tools, prompt=self.backstory)

    def creat_user_prompt(self, task: str, task_expected_output=None, context=None):
        """
        Based on the user query, it creates the user query and returns it. User prompt contains the task, expected output and the relevant context

        Parameters:
        -----------
        task: provided user query
        task_expected_output: format of the task output, if required
        context: context of the task for more clarity

        Returns:
        --------
        user prompt
        """
        user_prompt = f"""
        You are an AI agent. You are part of a team of agents working together to complete a task.
        I'm going to give you the task description enclosed in <task_description></task_description> tags. I'll also give
        you the available context from the other agents in <context></context> tags. If the context
        is not available, the <context></context> tags will be empty. You'll also receive the task
        expected output enclosed in <task_expected_output></task_expected_output> tags. With all this information
        you need to create the best possible response, always respecting the format as describe in
        <task_expected_output></task_expected_output> tags. If expected output is not available, just create
        a meaningful response to complete the task.

        <task_description>
        {task}
        </task_description>

        <task_expected_output>
        {task_expected_output}
        </task_expected_output>

        <context>
        {context}
        </context>

        Your response:
        """

        return user_prompt

    def run(self, Task, context=None, **kwargs):
        """
        This method triggers the invoke method within react agent to perform the task.

        Parameters:
        -----------
        Task: instance of the Task class
        context: context of the task for more clarity
        **kwargs: dynamic input parameter with keys and values

        Returns:
        --------
        llm_response
        """
        task = Task.task_description.format(**kwargs)
        task_expected_ouptut = Task.task_output
        context = context

        user_prompt = self.creat_user_prompt(task, task_expected_ouptut, context)
        llm_response = self.react_agent.invoke(user_prompt)
        return llm_response
