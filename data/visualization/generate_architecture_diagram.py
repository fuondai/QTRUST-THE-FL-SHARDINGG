#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def ensure_output_dir():
    """Create output directory for diagrams."""
    os.makedirs('docs/architecture', exist_ok=True)
    os.makedirs('docs/exported_charts', exist_ok=True)

def create_architecture_diagram():
    """Create the architecture diagram for QTrust system."""
    ensure_output_dir()
    
    # Create figure
    plt.figure(figsize=(16, 12))
    ax = plt.gca()
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes
    nodes = [
        ('QTrust', {'pos': (0, 0)}),
        ('BlockchainEnv', {'pos': (-2, -2)}),
        ('DQNAgents', {'pos': (0, -2)}),
        ('AdaptiveConsensus', {'pos': (2, -2)}),
        ('MADRAPIDRouter', {'pos': (-3, -4)}),
        ('HTDCM', {'pos': (-1, -4)}),
        ('FederatedLearning', {'pos': (1, -4)}),
        ('CachingSystem', {'pos': (3, -4)})
    ]
    
    G.add_nodes_from(nodes)
    
    # Define node positions, colors, and sizes
    pos = {node: data['pos'] for node, data in nodes}
    
    node_colors = {
        'QTrust': '#3498DB',
        'BlockchainEnv': '#2ECC71',
        'DQNAgents': '#9B59B6',
        'AdaptiveConsensus': '#F1C40F',
        'MADRAPIDRouter': '#E74C3C',
        'HTDCM': '#1ABC9C',
        'FederatedLearning': '#34495E',
        'CachingSystem': '#F39C12'
    }
    
    node_sizes = {
        'QTrust': 2500,
        'BlockchainEnv': 2000,
        'DQNAgents': 2000,
        'AdaptiveConsensus': 2000,
        'MADRAPIDRouter': 1800,
        'HTDCM': 1800,
        'FederatedLearning': 1800,
        'CachingSystem': 1800
    }
    
    widths = [2.5, 2.5, 2.5, 2.0, 2.0, 2.0, 2.0]
    heights = [1.2, 1.2, 1.2, 1.0, 1.0, 1.0, 1.0]
    
    # Draw nodes with different shapes (using rectangle shapes)
    node_groups = [
        ['QTrust'],
        ['BlockchainEnv', 'DQNAgents', 'AdaptiveConsensus'],
        ['MADRAPIDRouter', 'HTDCM', 'FederatedLearning', 'CachingSystem']
    ]
    
    # Draw nodes using different node types
    for group in node_groups:
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=group,
            node_color=[node_colors[n] for n in group],
            node_size=[node_sizes[n] for n in group],
            alpha=0.8,
            node_shape='s',  # Using 's' for square shape instead of 'rectangle'
            edgecolors='black',
            linewidths=2
        )
    
    # Add edges
    edges = [
        ('QTrust', 'BlockchainEnv'),
        ('QTrust', 'DQNAgents'),
        ('QTrust', 'AdaptiveConsensus'),
        ('BlockchainEnv', 'MADRAPIDRouter'),
        ('BlockchainEnv', 'HTDCM'),
        ('DQNAgents', 'FederatedLearning'),
        ('AdaptiveConsensus', 'CachingSystem'),
        ('MADRAPIDRouter', 'HTDCM'),
        ('HTDCM', 'FederatedLearning'),
        ('FederatedLearning', 'CachingSystem')
    ]
    
    G.add_edges_from(edges)
    
    # Draw edges with different styles
    nx.draw_networkx_edges(
        G, pos,
        edgelist=edges[:3],
        width=3,
        alpha=0.7,
        edge_color='#2C3E50',
        style='solid',
        arrowsize=20,
        arrowstyle='->'
    )
    
    nx.draw_networkx_edges(
        G, pos,
        edgelist=edges[3:7],
        width=2.5,
        alpha=0.7,
        edge_color='#7F8C8D',
        style='solid',
        arrowsize=15,
        arrowstyle='->'
    )
    
    nx.draw_networkx_edges(
        G, pos,
        edgelist=edges[7:],
        width=2,
        alpha=0.6,
        edge_color='#95A5A6',
        style='dashed',
        arrowsize=15,
        arrowstyle='->'
    )
    
    # Add labels
    nx.draw_networkx_labels(
        G, pos,
        font_size=14,
        font_family='sans-serif',
        font_weight='bold'
    )
    
    # Descriptions for each module
    descriptions = {
        'QTrust': 'Core Framework: Coordinates all component interactions',
        'BlockchainEnv': 'Blockchain Environment: Simulates network with sharding',
        'DQNAgents': 'DQN Agents: Implements Rainbow DQN for optimization',
        'AdaptiveConsensus': 'Adaptive Consensus: Selects optimal consensus protocol',
        'MADRAPIDRouter': 'MADRAPID Router: Smart transaction routing between shards',
        'HTDCM': 'HTDCM: Hierarchical Trust-based Data Center Mechanism',
        'FederatedLearning': 'Federated Learning: Privacy-preserving distributed training',
        'CachingSystem': 'Caching System: Optimizes access with intelligent strategies'
    }
    
    # Add descriptions below nodes
    for node, desc in descriptions.items():
        x, y = pos[node]
        plt.text(
            x, y-0.3,
            desc,
            fontsize=10,
            ha='center',
            va='center',
            color='black',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.5')
        )
    
    # Set plot limits and remove axis
    plt.xlim(-4.5, 4.5)
    plt.ylim(-5.5, 1.5)
    plt.axis('off')
    
    # Add title
    plt.title('QTrust Architecture: Component Relationship Diagram', fontsize=18, pad=20)
    
    # Add legend for layers
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#3498DB', 
                  markersize=15, label='Core Layer'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#2ECC71', 
                  markersize=15, label='Environment Layer'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#9B59B6', 
                  markersize=15, label='Intelligence Layer'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#F1C40F', 
                  markersize=15, label='Consensus Layer'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#E74C3C', 
                  markersize=15, label='Routing Layer'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#1ABC9C', 
                  markersize=15, label='Trust Layer'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#34495E', 
                  markersize=15, label='Learning Layer'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#F39C12', 
                  markersize=15, label='Caching Layer')
    ]
    
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.15), 
               ncol=4, fancybox=True, shadow=True)
    
    # Add watermark
    plt.figtext(0.5, 0.01, 'QTrust - Advanced Blockchain Research Framework', 
                ha='center', fontsize=10, color='gray')
    
    # Save figure
    plt.tight_layout()
    plt.savefig('docs/architecture/qtrust_architecture.png', dpi=300, bbox_inches='tight')
    plt.savefig('docs/exported_charts/architecture_diagram.png', dpi=300, bbox_inches='tight')
    
    print("Architecture diagram created: docs/architecture/qtrust_architecture.png")
    print("Architecture diagram copied to: docs/exported_charts/architecture_diagram.png")

def create_system_overview():
    """Tạo biểu đồ tổng quan hệ thống QTrust."""
    # Tạo figure
    plt.figure(figsize=(14, 8))
    
    # Tạo vùng trung tâm cho 'QTrust Core'
    core_box = plt.Rectangle((-4, -3), 8, 6, fill=True, alpha=0.1, 
                             facecolor='#3498db', edgecolor='#2980b9', lw=2)
    plt.gca().add_patch(core_box)
    plt.text(0, 2.5, 'QTrust Core', fontsize=18, fontweight='bold', 
             horizontalalignment='center', color='#2980b9')
    
    # Các module chính
    modules = [
        {'name': 'Blockchain Environment', 'pos': (-2.5, 1), 'color': '#e74c3c'},
        {'name': 'Deep Reinforcement Learning', 'pos': (2.5, 1), 'color': '#2ecc71'},
        {'name': 'Federated Learning', 'pos': (-2.5, -1), 'color': '#34495e'},
        {'name': 'HTDCM Security', 'pos': (2.5, -1), 'color': '#1abc9c'},
    ]
    
    # Vẽ các module chính
    for module in modules:
        x, y = module['pos']
        rect = plt.Rectangle((x-1.5, y-0.6), 3, 1.2, fill=True, alpha=0.7,
                            facecolor=module['color'], edgecolor='black', lw=1.5)
        plt.gca().add_patch(rect)
        plt.text(x, y, module['name'], fontsize=12, fontweight='bold',
                horizontalalignment='center', verticalalignment='center', color='white')
    
    # Các thành phần bên ngoài
    external_components = [
        {'name': 'Transaction Pool', 'pos': (-7, 1.5), 'color': '#f39c12'},
        {'name': 'Consensus Protocols', 'pos': (-7, -1.5), 'color': '#9b59b6'},
        {'name': 'Shards', 'pos': (7, 1.5), 'color': '#7f8c8d'},
        {'name': 'Network Nodes', 'pos': (7, -1.5), 'color': '#16a085'},
    ]
    
    # Vẽ các thành phần bên ngoài
    for comp in external_components:
        x, y = comp['pos']
        rect = plt.Rectangle((x-1.5, y-0.6), 3, 1.2, fill=True, alpha=0.7,
                            facecolor=comp['color'], edgecolor='black', lw=1.5)
        plt.gca().add_patch(rect)
        plt.text(x, y, comp['name'], fontsize=12, fontweight='bold',
                horizontalalignment='center', verticalalignment='center', color='white')
    
    # Vẽ các mũi tên kết nối
    arrows = [
        # External -> Core
        {'start': (-5.5, 1.5), 'end': (-3.5, 1), 'color': '#f39c12'},
        {'start': (-5.5, -1.5), 'end': (-3.5, -1), 'color': '#9b59b6'},
        {'start': (5.5, 1.5), 'end': (3.5, 1), 'color': '#7f8c8d'},
        {'start': (5.5, -1.5), 'end': (3.5, -1), 'color': '#16a085'},
        
        # Core internal
        {'start': (-1.5, 1), 'end': (1, 1), 'color': '#3498db'},
        {'start': (-1.5, -1), 'end': (1, -1), 'color': '#3498db'},
        {'start': (2.5, 0.4), 'end': (2.5, -0.4), 'color': '#3498db'},
        {'start': (-2.5, 0.4), 'end': (-2.5, -0.4), 'color': '#3498db'},
    ]
    
    # Vẽ các mũi tên
    for arrow in arrows:
        plt.annotate('', xy=arrow['end'], xytext=arrow['start'],
                    arrowprops=dict(arrowstyle='->', lw=2, color=arrow['color']))
    
    # Thêm khung viền cho toàn bộ hệ thống
    system_box = plt.Rectangle((-8, -4), 16, 8, fill=False, 
                              edgecolor='#2c3e50', lw=3, linestyle='--')
    plt.gca().add_patch(system_box)
    
    # Thêm tiêu đề
    plt.title('QTrust System Architecture Overview', fontsize=22, fontweight='bold', pad=20)
    
    # Thêm note
    note = "QTrust: DRL and Federated Learning-based blockchain sharding solution"
    plt.figtext(0.5, 0.01, note, wrap=True, horizontalalignment='center', fontsize=12)
    
    # Điều chỉnh layout
    plt.axis('off')
    plt.xlim(-9, 9)
    plt.ylim(-4.5, 4)
    plt.tight_layout()
    
    # Lưu hình
    plt.savefig('docs/exported_charts/system_overview.png', dpi=300, bbox_inches='tight')
    print("System overview diagram created: docs/exported_charts/system_overview.png")
    plt.close()

def main():
    """Run the architecture diagram generation."""
    create_architecture_diagram()
    create_system_overview()
    print("Architecture diagram generation completed.")

if __name__ == "__main__":
    main() 