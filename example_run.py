"""
SquadAI examples — simple to advanced.

Run all demos:
    python example_run.py

Run a single demo:
    python example_run.py --demo 1
    python example_run.py --demo 2
    python example_run.py --demo 3
"""

import argparse
from pprint import pprint

from squadAI.tools import tool_wrapper
from squadAI.task import Task
from squadAI.createAgent import Agent
from squadAI.squadAgent import SquadAgents

TOOL_RULES = (
    "You MUST use your tools for every calculation or lookup — never guess numbers. "
    "After tools return, give your final answer inside <response>...</response> tags."
)

PRODUCT_CATALOG = {
    "laptop": 999.00,
    "monitor": 349.00,
    "keyboard": 79.00,
}

TAX_RATES = {
    "CA": 0.0875,
    "TX": 0.0825,
    "NY": 0.08875,
}


# ---------------------------------------------------------------------------
# Shared tools
# ---------------------------------------------------------------------------


@tool_wrapper
def add_two_numbers(first_number: float, second_number: float) -> float:
    """Add two numbers."""
    return first_number + second_number


@tool_wrapper
def multiply_two_numbers(first_number: float, second_number: float) -> float:
    """Multiply two numbers."""
    return first_number * second_number


@tool_wrapper
def lookup_product_price(product_name: str) -> float:
    """Look up the unit price of a product. Valid products: laptop, monitor, keyboard."""
    key = product_name.strip().lower()
    if key not in PRODUCT_CATALOG:
        raise ValueError(f"Unknown product '{product_name}'. Try: laptop, monitor, keyboard.")
    return PRODUCT_CATALOG[key]


@tool_wrapper
def lookup_tax_rate(state: str) -> float:
    """Look up the sales tax rate for a US state (two-letter code). Valid: CA, TX, NY."""
    key = state.strip().upper()
    if key not in TAX_RATES:
        raise ValueError(f"Unknown state '{state}'. Try: CA, TX, NY.")
    return TAX_RATES[key]


# ---------------------------------------------------------------------------
# Demo 1: single agent, no tools
# ---------------------------------------------------------------------------


def demo_physics():
    print("=" * 60)
    print("Demo 1: Physics question (no tools)")
    print("=" * 60)

    physics_agent = Agent(
        backstory="You are a physics teacher. Explain concepts simply.",
    )
    physics_task = Task(
        task_description="In one short paragraph, what happens when you push a box on the floor?",
        agent=physics_agent,
    )
    squad = SquadAgents(agents=[physics_agent], tasks=[physics_task])
    pprint(squad.run())
    print()


# ---------------------------------------------------------------------------
# Demo 2: two agents, tools + linear task chaining  ->  (2 + 3) * 4 = 20
# ---------------------------------------------------------------------------


def demo_math_chain(a: float = 2, b: float = 3, c: float = 4):
    print("=" * 60)
    print(f"Demo 2: ({a} + {b}) * {c} via tools and dependencies")
    print("=" * 60)

    add_agent = Agent(
        backstory=f"You are a math expert. {TOOL_RULES}",
        tools=[add_two_numbers],
        max_iterations=8,
    )
    multiply_agent = Agent(
        backstory=f"You are a math expert. {TOOL_RULES}",
        tools=[multiply_two_numbers],
        max_iterations=8,
    )

    task1 = Task(task_description="Add {a} and {b}", agent=add_agent)
    task2 = Task(
        task_description="Multiply the previous result by {c}",
        dependency=[task1],
        agent=multiply_agent,
    )

    squad = SquadAgents(agents=[add_agent, multiply_agent], tasks=[task1, task2])
    pprint(squad.run(a=a, b=b, c=c))
    print()


# ---------------------------------------------------------------------------
# Demo 3: five agents, branching dependencies — purchase approval pipeline
# ---------------------------------------------------------------------------


def demo_purchase_approval(
    quantity: int = 3,
    product: str = "laptop",
    state: str = "CA",
    budget: int = 3000,
):
    print("=" * 60)
    print("Demo 3: Purchase approval pipeline (multi-agent, branching deps)")
    print("=" * 60)
    print(f"Order: {quantity} x {product}  |  State: {state}  |  Budget: ${budget:,}")
    print("-" * 60)

    catalog_agent = Agent(
        backstory=f"You are a product catalog specialist. {TOOL_RULES}",
        tools=[lookup_product_price],
        max_iterations=8,
    )
    tax_agent = Agent(
        backstory=f"You are a tax compliance specialist. {TOOL_RULES}",
        tools=[lookup_tax_rate],
        max_iterations=8,
    )
    subtotal_agent = Agent(
        backstory=f"You are a billing clerk. {TOOL_RULES} "
        "Extract the unit price from <context> and multiply by the quantity.",
        tools=[multiply_two_numbers],
        max_iterations=8,
    )
    checkout_agent = Agent(
        backstory=f"You are a checkout cashier. {TOOL_RULES} "
        "From <context>: extract the subtotal and the tax rate. "
        "First multiply subtotal by tax rate to get tax amount, "
        "then add subtotal and tax amount for the final total. "
        "State the final total clearly in your response.",
        tools=[multiply_two_numbers, add_two_numbers],
        max_iterations=10,
    )
    budget_agent = Agent(
        backstory=(
            "You are a finance approver. Read the final order total from <context> "
            "and compare it to the budget given in the task. "
            "Reply with APPROVED if total <= budget, otherwise REJECTED, "
            "and briefly explain why. Use <response>...</response> tags."
        ),
        max_iterations=4,
    )

    task_price = Task(
        task_description="Look up the unit price for product '{product}'.",
        agent=catalog_agent,
    )
    task_tax = Task(
        task_description="Look up the sales tax rate for state '{state}'.",
        agent=tax_agent,
    )
    task_subtotal = Task(
        task_description=(
            "Multiply the unit price from context by quantity {quantity} to get the subtotal."
        ),
        dependency=[task_price],
        agent=subtotal_agent,
    )
    task_total = Task(
        task_description=(
            "Calculate the final order total including tax using the subtotal "
            "and tax rate from context."
        ),
        dependency=[task_tax, task_subtotal],
        agent=checkout_agent,
    )
    task_approval = Task(
        task_description=(
            "Compare the final order total from context against budget ${budget}. "
            "Respond APPROVED or REJECTED."
        ),
        dependency=[task_total],
        agent=budget_agent,
    )

    squad = SquadAgents(
        agents=[catalog_agent, tax_agent, subtotal_agent, checkout_agent, budget_agent],
        tasks=[task_price, task_tax, task_subtotal, task_total, task_approval],
    )

    result = squad.run(quantity=quantity, product=product, state=state, budget=budget)

    print("-" * 60)
    print("Final decision:")
    pprint(result)
    print()


DEMOS = {
    1: demo_physics,
    2: demo_math_chain,
    3: demo_purchase_approval,
}


def main():
    parser = argparse.ArgumentParser(description="Run SquadAI example demos.")
    parser.add_argument(
        "--demo",
        type=int,
        choices=DEMOS,
        help="Run a single demo by number (1-3). Omit to run all.",
    )
    args = parser.parse_args()

    if args.demo:
        DEMOS[args.demo]()
    else:
        for run in DEMOS.values():
            run()


if __name__ == "__main__":
    main()
