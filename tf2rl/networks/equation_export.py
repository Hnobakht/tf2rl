"""Exact inference graphs with zero-edge removal, constant folding and shared terms.

The JSON graph and generated NumPy module do not require TensorFlow. This exports
implemented operators, not a symbolic approximation or a rounded coefficient fit.
"""
import inspect
import json
from pathlib import Path

import numpy as np


def evaluate_equation(graph, inputs):
    """Evaluate a version-1 graph on a batch of observations (and actions)."""
    if graph['version'] != 1:
        raise ValueError('Unsupported equation graph version')
    inputs = np.asarray(inputs, dtype=np.float32)
    if inputs.ndim != 2 or inputs.shape[1] != len(graph['inputs']):
        raise ValueError('Expected a batch with one column per named input')
    values = []
    for node in graph['nodes']:
        op = node['op']
        args = [values[i] for i in node.get('args', [])]
        if op == 'input':
            value = inputs[:, node['index']]
        elif op == 'constant':
            value = np.float32(node['value'])
        elif op == 'affine':
            columns = np.stack(np.broadcast_arrays(*args), axis=-1)
            value = columns @ np.asarray(node['weights'], np.float32) + np.float32(node['bias'])
        elif op == 'sin':
            value = np.sin(args[0])
        elif op == 'cos':
            value = np.cos(args[0])
        elif op == 'sigmoid':
            # Equivalent stable sigmoid, avoiding exp overflow for negative logits.
            z = np.exp(-np.abs(args[0]))
            value = np.where(args[0] >= 0, 1 / (1 + z), z / (1 + z))
        elif op == 'log':
            value = np.log(args[0])
        elif op == 'square':
            value = np.square(args[0])
        elif op == 'add':
            value = args[0] + args[1]
        elif op == 'subtract':
            value = args[0] - args[1]
        elif op == 'divide':
            value = args[0] / args[1]
        elif op == 'maximum':
            value = np.maximum(args[0], args[1])
        else:
            raise ValueError(f'Unsupported operator: {op}')
        values.append(np.asarray(value, dtype=np.float32))
    return {name: np.broadcast_to(values[index], (len(inputs),)).reshape(-1, 1).copy()
            for name, index in graph['outputs'].items()}


class _Graph:
    def __init__(self, inputs):
        self.inputs = inputs
        self.nodes = []
        self.cache = {}

    def add(self, op, args=(), **attributes):
        node = dict(op=op, **attributes)
        if args:
            node['args'] = list(args)
        if op not in ('input', 'constant') and all(self.nodes[i]['op'] == 'constant' for i in args):
            graph = {'version': 1, 'inputs': self.inputs, 'nodes': self.nodes + [node],
                         'outputs': {'value': len(self.nodes)}}
            value = evaluate_equation(graph, np.zeros((1, len(self.inputs)), np.float32))['value'].item()
            return self.constant(value)
        key = json.dumps(node, sort_keys=True, allow_nan=False)
        if key not in self.cache:
            self.cache[key] = len(self.nodes)
            self.nodes.append(node)
        return self.cache[key]

    def constant(self, value):
        return self.add('constant', value=float(np.float32(value)))

    def affine(self, inputs, weights, bias):
        selected = [(node, float(weight)) for node, weight in zip(inputs, weights) if weight != 0]
        if not selected:
            return self.constant(bias)
        args, weights = zip(*selected)
        if len(args) == 1 and weights[0] == 1 and bias == 0:
            return args[0]
        return self.add('affine', args, weights=weights, bias=float(bias))

    def finish(self, outputs, metadata):
        reachable = set()

        def visit(index):
            if index not in reachable:
                reachable.add(index)
                for parent in self.nodes[index].get('args', []):
                    visit(parent)

        for index in outputs.values():
            visit(index)
        indices = sorted(reachable)
        mapping = {old: new for new, old in enumerate(indices)}
        nodes = []
        for index in indices:
            node = self.nodes[index].copy()
            if 'args' in node:
                node['args'] = [mapping[i] for i in node['args']]
            nodes.append(node)
        return {'version': 1, 'inputs': self.inputs, 'nodes': nodes,
                    'outputs': {name: mapping[i] for name, i in outputs.items()}, 'metadata': metadata}


def extract_equation(model, *, reward_kind, input_names=None, metadata=None):
    """Export EQN-GAIL log odds or EQN-AIRL's raw reward in inference mode.

    Dropout is inactive at inference. Singular penalties and AIRL's separate dense
    shaping potential are training components and are not part of this expression.
    """
    if reward_kind not in ('gail', 'airl'):
        raise ValueError('reward_kind must be gail or airl')
    width = model.parameter_layers[0].w.shape[0]
    names = list(input_names) if input_names is not None else [f'x_{i}' for i in range(width)]
    if len(names) != width or len(set(names)) != width:
        raise ValueError('Input names must be unique and match the network input width')
    graph = _Graph(names)
    features = [graph.add('input', index=i) for i in range(width)]
    zero, relaxation = graph.constant(0), graph.constant(.1)
    for layer in model.parameter_layers[:-1]:
        w, b = [value.numpy() for value in layer.effective_parameters()]
        affine = [graph.affine(features, w[:, i], b[i]) for i in range(len(b))]
        features = []
        for group in range(layer.v):
            index = group * layer.input_width
            for op, size in layer.operators:
                a = affine[index]
                if op == 'id':
                    value = a
                elif op in ('sin', 'cos', 'sig'):
                    value = graph.add('sigmoid' if op == 'sig' else op, (a,))
                elif op == 'log':
                    value = graph.add('log', (graph.add('add', (
                        graph.add('maximum', (a, zero)), relaxation)),))
                elif op == 'mult':
                    value = graph.add('square', (graph.add('add', (a, affine[index + 1])),))
                else:
                    denominator = graph.add('add', (graph.add('maximum', (affine[index + 1], zero)), relaxation))
                    value = graph.add('divide', (graph.add('add', (a, relaxation)), denominator))
                features.append(value)
                index += size
    w, b = [value.numpy() for value in model.layer_out.effective_parameters()]
    raw = graph.affine(features, w[:, 0], b[0])
    activation = getattr(model.layer_out.activation, '__name__', None)
    expected = 'sigmoid' if reward_kind == 'gail' else 'linear'
    if activation != expected:
        raise ValueError(f'{reward_kind} export requires {expected} output, got {activation}')
    outputs = {'raw_output': raw}
    if reward_kind == 'gail':
        probability = graph.add('sigmoid', (raw,))
        epsilon = graph.constant(1e-8)
        log_real = graph.add('log', (graph.add('add', (probability, epsilon)),))
        log_fake = graph.add('log', (graph.add('add', (
            graph.add('subtract', (graph.constant(1), probability)), epsilon)),))
        outputs.update(probability=probability, reward=graph.add('subtract', (log_real, log_fake)))
    else:
        outputs['reward'] = raw
    details = dict(metadata or {})
    details.update(reward_kind=reward_kind, inference=True, coefficient_dtype='float32',
                   sparsity={key: float(np.asarray(value)) for key, value in model.sparsity_metrics().items()})
    return graph.finish(outputs, details)


def format_equation(graph):
    """Readable shared expressions; no exponential expansion or coefficient rounding."""
    lines = ['# Inference equation; float32 coefficients; dropout disabled.',
             '# Input order: ' + ', '.join(graph['inputs'])]
    for index, node in enumerate(graph['nodes']):
        op = node['op']
        args = [f't_{i}' for i in node.get('args', [])]
        if op == 'input':
            expression = graph['inputs'][node['index']]
        elif op == 'constant':
            expression = repr(node['value'])
        elif op == 'affine':
            expression = ' + '.join(f'({weight!r} * {arg})' for weight, arg in zip(node['weights'], args))
            if node['bias'] != 0:
                expression += f" + ({node['bias']!r})"
        elif op in ('add', 'subtract', 'divide'):
            symbol = {'add': '+', 'subtract': '-', 'divide': '/'}[op]
            expression = f'({args[0]} {symbol} {args[1]})'
        else:
            expression = f"{op}({', '.join(args)})"
        lines.append(f't_{index} = {expression}')
    lines.extend(f'{name} = t_{index}' for name, index in graph['outputs'].items())
    return '\n'.join(lines) + '\n'


def save_equation(graph, path):
    """Write JSON, readable expressions and a standalone NumPy evaluator."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix('.json').write_text(json.dumps(graph, indent=2, allow_nan=False) + '\n')
    path.with_suffix('.txt').write_text(format_equation(graph))
    source = ('"""Exported inference equation. Call predict(batch) for named outputs."""\n'
              'import numpy as np\n\n' + inspect.getsource(evaluate_equation)
              + '\nGRAPH = ' + repr(graph) + '\n\ndef predict(inputs):\n'
              '    return evaluate_equation(GRAPH, inputs)\n')
    path.with_suffix('.py').write_text(source)
