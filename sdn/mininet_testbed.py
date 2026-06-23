"""Mininet testbed builder for SDN validation of Phase-1 reactive TE.

Creates Mininet topologies matching the simulation datasets (Abilene, GEANT, etc.)
and generates iperf traffic matching the TM for end-to-end validation.

Requirements: mininet, ryu (pip install ryu), openvswitch
Run with: sudo python -m sdn.mininet_testbed --topology abilene
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MininetHost:
    name: str
    ip: str
    mac: str


@dataclass
class MininetSwitch:
    name: str
    dpid: str


@dataclass
class MininetLink:
    src: str
    dst: str
    src_port: int
    dst_port: int
    bw_mbps: float
    delay_ms: float = 1.0


@dataclass
class MininetTopology:
    """Complete Mininet topology specification."""
    hosts: List[MininetHost] = field(default_factory=list)
    switches: List[MininetSwitch] = field(default_factory=list)
    links: List[MininetLink] = field(default_factory=list)
    controller_ip: str = "127.0.0.1"
    controller_port: int = 6633


def build_mininet_topology(
    nodes: Sequence[str],
    edges: Sequence[Tuple[str, str]],
    capacities: np.ndarray,
    weights: np.ndarray,
) -> MininetTopology:
    """Convert simulation topology to Mininet specification.

    Creates one OVS switch per node and one host per node.
    Links between switches mirror the simulation edges with matching capacities.
    """
    topo = MininetTopology()

    # One switch + one host per node
    for i, node in enumerate(nodes):
        sw_name = f"s{i+1}"
        host_name = f"h{i+1}"
        dpid = f"{i+1:016x}"
        ip = f"10.0.{i+1}.1/24"
        mac = f"00:00:00:00:{i+1:02x}:01"

        topo.switches.append(MininetSwitch(name=sw_name, dpid=dpid))
        topo.hosts.append(MininetHost(name=host_name, ip=ip, mac=mac))

        # Host-to-switch link (high bandwidth, no bottleneck)
        topo.links.append(MininetLink(
            src=host_name, dst=sw_name,
            src_port=0, dst_port=100 + i,
            bw_mbps=10000.0,
        ))

    # Switch-to-switch links from edges
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    port_counter: Dict[str, int] = {f"s{i+1}": 1 for i in range(len(nodes))}

    for edge_idx, (src, dst) in enumerate(edges):
        src_sw = f"s{node_to_idx[src]+1}"
        dst_sw = f"s{node_to_idx[dst]+1}"
        src_port = port_counter[src_sw]
        dst_port = port_counter[dst_sw]
        port_counter[src_sw] += 1
        port_counter[dst_sw] += 1

        bw = float(capacities[edge_idx])
        delay = float(weights[edge_idx])

        topo.links.append(MininetLink(
            src=src_sw, dst=dst_sw,
            src_port=src_port, dst_port=dst_port,
            bw_mbps=bw,
            delay_ms=delay,
        ))

    return topo


def generate_mininet_script(topo: MininetTopology, output_path: str | Path) -> Path:
    """Generate a standalone Python script that builds and runs the Mininet topology."""
    output_path = Path(output_path)

    lines = [
        '#!/usr/bin/env python3',
        '"""Auto-generated Mininet topology for Phase-1 TE validation."""',
        '',
        'from mininet.net import Mininet',
        'from mininet.node import OVSSwitch, RemoteController',
        'from mininet.link import TCLink',
        'from mininet.cli import CLI',
        'from mininet.log import setLogLevel',
        '',
        '',
        'def build_network():',
        '    setLogLevel("info")',
        f'    net = Mininet(controller=RemoteController, switch=OVSSwitch, link=TCLink)',
        '',
        f'    # Controller at {topo.controller_ip}:{topo.controller_port}',
        f'    c0 = net.addController("c0", ip="{topo.controller_ip}", port={topo.controller_port})',
        '',
        '    # Switches',
    ]

    for sw in topo.switches:
        lines.append(f'    {sw.name} = net.addSwitch("{sw.name}", dpid="{sw.dpid}", protocols="OpenFlow13")')

    lines.append('')
    lines.append('    # Hosts')
    for host in topo.hosts:
        lines.append(f'    {host.name} = net.addHost("{host.name}", ip="{host.ip}", mac="{host.mac}")')

    lines.append('')
    lines.append('    # Links')
    for link in topo.links:
        delay_str = f'{link.delay_ms}ms' if link.delay_ms > 0 else '0ms'
        lines.append(
            f'    net.addLink("{link.src}", "{link.dst}", '
            f'port1={link.src_port}, port2={link.dst_port}, '
            f'bw={link.bw_mbps}, delay="{delay_str}")'
        )

    lines.extend([
        '',
        '    net.start()',
        '    print("Mininet topology started. Use CLI or connect controller.")',
        '    CLI(net)',
        '    net.stop()',
        '',
        '',
        'if __name__ == "__main__":',
        '    build_network()',
    ])

    output_path.write_text('\n'.join(lines) + '\n')
    output_path.chmod(0o755)
    logger.info(f"Generated Mininet script: {output_path}")
    return output_path


# ── Traffic generation ──────────────────────────────────────────────────────

def generate_traffic_script(
    od_pairs: Sequence[Tuple[str, str]],
    tm_vector: np.ndarray,
    nodes: Sequence[str],
    duration_sec: int = 10,
    output_path: str | Path = "traffic_gen.sh",
) -> Path:
    """Generate iperf commands to replay a TM snapshot on Mininet.

    Each OD pair with demand > 0 gets an iperf UDP flow from host_src to host_dst.
    """
    output_path = Path(output_path)
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    lines = ['#!/bin/bash', '# Auto-generated traffic for Phase-1 TE validation', '']

    # Start iperf servers on all hosts
    lines.append('# Start iperf servers')
    for i in range(len(nodes)):
        host = f"h{i+1}"
        lines.append(f'{host} iperf -s -u -p 5001 &')
    lines.append('sleep 2')
    lines.append('')

    # Start iperf clients for each OD with demand
    lines.append('# Start traffic flows')
    for od_idx, (src, dst) in enumerate(od_pairs):
        demand = float(tm_vector[od_idx])
        if demand <= 0:
            continue

        src_idx = node_to_idx.get(src, 0)
        dst_idx = node_to_idx.get(dst, 0)
        src_host = f"h{src_idx+1}"
        dst_ip = f"10.0.{dst_idx+1}.1"

        # Convert demand to bandwidth string
        bw_str = f"{demand:.2f}M"
        lines.append(
            f'{src_host} iperf -c {dst_ip} -u -b {bw_str} -t {duration_sec} -p 5001 &'
        )

    lines.extend([
        '',
        f'echo "Traffic running for {duration_sec}s..."',
        f'sleep {duration_sec}',
        'echo "Done."',
    ])

    output_path.write_text('\n'.join(lines) + '\n')
    output_path.chmod(0o755)
    logger.info(f"Generated traffic script: {output_path}")
    return output_path


# ── Topology export for external tools ──────────────────────────────────────

def export_topology_json(
    nodes: Sequence[str],
    edges: Sequence[Tuple[str, str]],
    capacities: np.ndarray,
    weights: np.ndarray,
    od_pairs: Sequence[Tuple[str, str]],
    output_path: str | Path,
) -> Path:
    """Export topology as JSON for use with external SDN controllers."""
    output_path = Path(output_path)
    data = {
        "nodes": list(nodes),
        "edges": [{"src": s, "dst": d, "capacity": float(capacities[i]), "weight": float(weights[i])}
                  for i, (s, d) in enumerate(edges)],
        "od_pairs": [{"src": s, "dst": d} for s, d in od_pairs],
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "num_od_pairs": len(od_pairs),
    }
    output_path.write_text(json.dumps(data, indent=2))
    logger.info(f"Exported topology JSON: {output_path}")
    return output_path


# ── CLI entry point ─────────────────────────────────────────────────────────

def main():
    """CLI: generate Mininet topology from a dataset."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate Mininet testbed for Phase-1 TE")
    parser.add_argument("--topology", type=str, default="abilene",
                        choices=["abilene", "geant", "ebone", "sprintlink", "tiscali", "germany50"],
                        help="Topology to generate")
    parser.add_argument("--output-dir", type=str, default="sdn/generated",
                        help="Output directory for generated scripts")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from phase1_reactive.eval.common import load_phase1_datasets
    datasets = load_phase1_datasets()

    dataset = None
    for ds in datasets:
        if args.topology.lower() in ds.key.lower():
            dataset = ds
            break

    if dataset is None:
        print(f"Topology '{args.topology}' not found in datasets")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    topo = build_mininet_topology(
        dataset.nodes, dataset.edges,
        np.asarray(dataset.capacities), np.asarray(dataset.weights),
    )

    generate_mininet_script(topo, out_dir / f"mininet_{args.topology}.py")
    export_topology_json(
        dataset.nodes, dataset.edges,
        np.asarray(dataset.capacities), np.asarray(dataset.weights),
        dataset.od_pairs,
        out_dir / f"topology_{args.topology}.json",
    )

    # Generate traffic for first test timestep
    sp = dataset.split
    test_start = int(sp["test_start"])
    if test_start < dataset.tm.shape[0]:
        generate_traffic_script(
            dataset.od_pairs, dataset.tm[test_start], dataset.nodes,
            duration_sec=30,
            output_path=out_dir / f"traffic_{args.topology}.sh",
        )

    print(f"Generated Mininet testbed in {out_dir}/")


if __name__ == "__main__":
    main()
