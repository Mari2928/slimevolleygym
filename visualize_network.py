import graphviz
import matplotlib.pyplot as plt
import numpy as np

def visualize_network_with_activations(genome, config, node_names=None):
    dot = graphviz.Digraph(format='png')

    # Define inputs and add them to the graph
    inputs = config.genome_config.input_keys
    for i in inputs:
        name = node_names[i] if node_names else f"Input {i}"
        dot.node(name, shape='circle', style='filled', fillcolor='lightgray')

    # Define outputs and add them to the graph
    outputs = config.genome_config.output_keys
    for o in outputs:
        name = node_names[o] if node_names else f"Output {o}"

        # Here we use genome.nodes[o].activation to access the activation function
        label = f'{name}\n({genome.nodes[o].activation})'
        dot.node(name, label, shape='circle', style='filled', fillcolor='lightblue')

    # Add hidden nodes with activation function
    for node_id, node in genome.nodes.items():
        if node_id not in inputs and node_id not in outputs:
            name = node_names[node_id] if node_names else f"Hidden {node_id}"
            label = f'{name}\n({node.activation})'
            dot.node(name, label, shape='circle', style='filled', fillcolor='white')

    # Add edges with weights
    for cg in genome.connections.values():
        if cg.enabled:
            input_node = node_names[cg.key[0]] if node_names else str(cg.key[0])
            output_node = node_names[cg.key[1]] if node_names else str(cg.key[1])
            dot.edge(input_node, output_node, label='{:.2f}'.format(cg.weight))

    return dot


def create_node_names(genome, config):
    node_names = {}

    # Create names for the input nodes
    for i, input_key in enumerate(config.genome_config.input_keys):
        node_names[input_key] = f'Input{i + 1}'

    # Create names for the output nodes
    for i, output_key in enumerate(config.genome_config.output_keys):
        node_names[output_key] = f'Output{i + 1}'

    # Create names for hidden nodes
    hidden_nodes = [n for n in genome.nodes if n not in node_names]
    for i, hidden_key in enumerate(hidden_nodes):
        node_names[hidden_key] = f'Hidden{i + 1}'

    return node_names