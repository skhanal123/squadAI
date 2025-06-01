# This python file provide sample examples to run the squadAI

from pprint import pprint
from squadAI.tools import tool_wrapper
from squadAI.task import Task
from squadAI.createAgent import Agent
from squadAI.squadAgent import SquadAgents


#### Sample Examples ####


#### Simple Test 1: Just to check the functionality ####
@tool_wrapper
def add_two_numbers(first_number: float, second_number: float):
    """
    This function adds two numbers and return the output

    Args:
        first_number (float): First number to be added
        second_number (float): Second number to be added

    Returns:
        float: The final output after adding to two number
    """
    output = first_number + second_number

    return output


@tool_wrapper
def multiply_two_numbers(first_number: float, second_number: float):
    """
    This function mulitply two numbers and return the output

    Args:
        first_number (float): First number for multiplication
        second_number (float): Second number for multiplication

    Returns:
        float: The final output after multiplying two numbers
    """
    output = first_number * second_number

    return output


add_agent = Agent(
    backstory="You are an expert in mathematics and you are capable of solving simple to complex math problems",
    tools=[add_two_numbers],
)

test_task1 = Task(
    task_description="Please add two numbers {a} and {b}", agent=add_agent
)

multiply_agent = Agent(
    backstory="You are an expert in mathematics and you are capable of solving simple to complex math problems",
    tools=[multiply_two_numbers],
)

test_task2 = Task(
    task_description="Please multply the output of previous task1 provided in the context below by {c}",
    dependency=[test_task1],
    agent=multiply_agent,
)

# AIsquad1 = SquadAgents(
#     agents=[add_agent, multiply_agent], tasks=[test_task1, test_task2]
# )

# pprint(AIsquad1.run(**{"a": 2, "b": 3, "c": 4}))

# -----Output-----#
"""
<thought>I need to add the numbers 2 and 3 using the available tool.</thought>
<tool_call>
{"name": "add_two_numbers", "arguments": {"first_number": 2, "second_number": 3}, "id": 0}
</tool_call>
<thought>I need to add the numbers 2 and 3 using the available tool.</thought>
<tool_call>
{"name": "add_two_numbers", "arguments": {"first_number": 2, "second_number": 3}, "id": 0}
</tool_call>

{"name": "add_two_numbers", "arguments": {"first_number": 2, "second_number": 3}, "id": 0}

<response>The sum of 2 and 3 is 5.</response>
<thought>The context provides the output of a previous task, which is the sum of 2 and 3, resulting in 5. The current task is to multiply this output by 4.</thought>
<tool_call>
{"name": "multiply_two_numbers", "arguments": {"first_number": 5, "second_number": 4}, "id": 0}
</tool_call>
<thought>The context provides the output of a previous task, which is the sum of 2 and 3, resulting in 5. The current task is to multiply this output by 4.</thought>
<tool_call>
{"name": "multiply_two_numbers", "arguments": {"first_number": 5, "second_number": 4}, "id": 0}
</tool_call>

{"name": "multiply_two_numbers", "arguments": {"first_number": 5, "second_number": 4}, "id": 0}

<response>The result of multiplying the previous task's output (5) by 4 is 20.</response>
"The result of multiplying the previous task's output (5) by 4 is 20."
"""


#### Simple Test 2: Just to check the functionality when tools are required to perform the task ####

physics_instructor = Agent(
    backstory="You are an expert in physics and you are capable of explaining the concepts in a simple language",
)

physics_task = Task(
    task_description="What happens when we apply force to an object?",
    agent=physics_instructor,
)

physics_AIsquad = SquadAgents(agents=[physics_instructor], tasks=[physics_task])

pprint(physics_AIsquad.run())

# -----Output-----#
"""
('When we apply force to an object, several outcomes can occur depending on '
 "the magnitude and direction of the force, as well as the object's properties "
 '(like mass and friction). Here’s what typically happens:'
 ''
 '1. **Object at Rest**:  '
 '   - If the applied force overcomes static friction (for objects on a '
 'surface), the object will start moving.  '
 '   - Newton’s First Law (Inertia) states that an object remains at rest '
 'unless acted upon by an unbalanced force.'
 ''
 '2. **Object in Motion**:  '
 '   - A force can accelerate the object (increase its speed), decelerate it '
 '(reduce speed), or change its direction (e.g., circular motion).  '
 '   - Newton’s Second Law states: \\( F = ma \\) (Force = mass × '
 'acceleration).  '
 ''
 '3. **Deformation**:  '
 '   - If the force exceeds the object’s structural integrity, it may bend, '
 'stretch, or break (e.g., squashing a sponge or snapping a rope).  '
 ''
 '4. **Rotation (Torque)**:  '
 '   - If the force is applied off-center, it can cause the object to rotate '
 '(e.g., turning a wrench).  '
 ''
 ''
 '3. **Deformation**:  '
 '   - If the force exceeds the object’s structural integrity, it may bend, '
 'stretch, or break (e.g., squashing a sponge or snapping a rope).  '
 ''
 '4. **Rotation (Torque)**:  '
 '   - If the force is applied off-center, it can cause the object to rotate '
 '(e.g., turning a wrench).  '
 ''
 '   - If the force exceeds the object’s structural integrity, it may bend, '
 'stretch, or break (e.g., squashing a sponge or snapping a rope).  '
 ''
 '4. **Rotation (Torque)**:  '
 '   - If the force is applied off-center, it can cause the object to rotate '
 '(e.g., turning a wrench).  '
 ''
 ''
 '4. **Rotation (Torque)**:  '
 '   - If the force is applied off-center, it can cause the object to rotate '
 '(e.g., turning a wrench).  '
 ''
 '4. **Rotation (Torque)**:  '
 '   - If the force is applied off-center, it can cause the object to rotate '
 '(e.g., turning a wrench).  '
 ''
 '(e.g., turning a wrench).  '
 ''
 '5. **Equilibrium**:  '
 '5. **Equilibrium**:  '
 '   - If opposing forces balance out (e.g., gravity and normal force on a '
 'table), the object’s state of motion won’t change.  '
 '   - If opposing forces balance out (e.g., gravity and normal force on a '
 'table), the object’s state of motion won’t change.  '
 'table), the object’s state of motion won’t change.  '
 ''
 '**Example**: Pushing a box:  '
 '- A small force might not move it (friction balances your push).  '
 ''
 '**Example**: Pushing a box:  '
 '- A small force might not move it (friction balances your push).  '
 '- A small force might not move it (friction balances your push).  '
 '- A stronger force accelerates it.  '
 '- An extreme force could tip or break it.  '
 ''
 'Would you like a deeper explanation of a specific scenario?')
"""
