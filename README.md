# NV Center in Diamond: Interactive Educational Notebooks

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/E-zClap/nv-center-notebook/main)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**A comprehensive, interactive educational resource for understanding Nitrogen-Vacancy (NV) centers in diamond from first principles.**

This collection of Jupyter notebooks provides a complete journey through the physics of NV centers, from basic atomic structure to advanced quantum sensing applications. Perfect for students, researchers, and practitioners working with quantum sensors and solid-state quantum systems.

## 🎯 What You'll Learn

- **Fundamental Physics**: Electronic structure, symmetry, and molecular orbital theory
- **Energy Level Construction**: Step-by-step derivation of the complete NV energy diagram
- **Quantum Sensing**: How NV centers become nanoscale magnetometers and thermometers
- **Practical Applications**: Real-world sensing protocols and experimental considerations
- **Interactive Visualizations**: 3D crystal structures, energy diagrams, and dynamic simulations

## 🚀 Quick Start

### Option 1: Run Online (Recommended for Beginners)
Click here to launch in your browser: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/E-zClap/nv-center-notebook/main)

### Option 2: Local Installation
```bash
git clone https://github.com/E-zClap/nv-center-notebook.git
cd nv-center-notebook
pip install -r requirements.txt
jupyter lab
```

## 📚 Learning Path

### 🎓 **Level 1: Fundamentals**
Start here if you're new to NV centers or need a refresher on quantum mechanics:

1. **[`nv_center_visualization.ipynb`](nv_center_visualization.ipynb)**
   - Interactive 3D visualization of the NV center in diamond lattice
   - Crystal structure and defect geometry
   - Perfect for building intuition

2. **[`energy_diagram/nv_01_center_fundamentals.ipynb`](energy_diagram/nv_01_center_fundamentals.ipynb)**
   - Defect geometry and electron counting
   - Molecular orbital construction from symmetry
   - Foundation for all NV physics

### 🔬 **Level 2: Energy Diagram Construction**
Deep dive into the electronic structure and energy levels:

3. **[`energy_diagram/nv_02_electronic_configuration_terms.ipynb`](energy_diagram/nv_02_electronic_configuration_terms.ipynb)**
   - Term symbols and electronic configurations
   - Ground and excited state terms

4. **[`energy_diagram/3_zero_field_splitting.ipynb`](energy_diagram/3_zero_field_splitting.ipynb)**
   - Origin of the famous 2.87 GHz ground state splitting
   - Spin-spin interactions and crystal field effects

5. **[`energy_diagram/4_excited_states_optical_transitions.ipynb`](energy_diagram/4_excited_states_optical_transitions.ipynb)**
   - Optical excitation cycle (green pump, red fluorescence)
   - Selection rules and transition probabilities

### ⚡ **Level 3: Advanced Physics**
Explore the mechanisms that make NV centers powerful quantum sensors:

6. **[`energy_diagram/6_singlet_states_intersystem_crossing.ipynb`](energy_diagram/6_singlet_states_intersystem_crossing.ipynb)**
   - Spin-selective intersystem crossing
   - Origin of optical spin polarization

7. **[`energy_diagram/7_optical_dynamics_spin_polarization.ipynb`](energy_diagram/7_optical_dynamics_spin_polarization.ipynb)**
   - Complete optical pumping cycle
   - Spin-dependent fluorescence readout

8. **[`energy_diagram/8_external_field_effects.ipynb`](energy_diagram/8_external_field_effects.ipynb)**
   - Magnetic field effects (Zeeman splitting)
   - Electric field and strain interactions
   - Basis for quantum sensing

### 🔧 **Level 4: Applications & Practice**
Apply your knowledge to real sensing scenarios:

9. **[`nv_center_energy_diagram.ipynb`](nv_center_energy_diagram.ipynb)**
   - Complete energy diagram with all levels
   - Experimental cycle and measurement protocols

10. **[`nv_center_symmetry_explorer.ipynb`](nv_center_symmetry_explorer.ipynb)**
    - Interactive exploration of symmetry effects
    - Group theory applications

## 🎯 Key Features

### 📊 **Interactive Visualizations**
- **3D Crystal Structure**: Rotate and explore the diamond lattice with NV defects
- **Energy Level Diagrams**: Dynamic plots showing how external fields affect energy levels
- **Symmetry Explorer**: Interactive group theory and molecular orbital visualizations

### 🧮 **From First Principles**
- **No "magic" formulas**: Every energy level and transition is derived step-by-step
- **Clear Physical Reasoning**: Understand *why* each effect occurs, not just *what* happens
- **Mathematical Rigor**: Complete derivations with proper symmetry analysis

### 🔬 **Practical Applications**
- **Sensing Protocols**: Learn how NV centers measure magnetic fields, temperature, and electric fields
- **Experimental Parameters**: Real values used in current research
- **Troubleshooting Guide**: Common issues and how to address them

## 🛠️ Technologies Used

- **Python 3.8+** with scientific computing stack
- **Jupyter Lab/Notebook** for interactive learning
- **Plotly** for 3D visualizations and interactive plots
- **NumPy/Matplotlib** for numerical calculations and 2D plots
- **ipywidgets** for interactive controls
- **ASE** (Atomic Simulation Environment) for crystal structures

## 📖 Prerequisites

### Required Background
- **Quantum Mechanics**: Basic understanding of wavefunctions, operators, and eigenvalue problems
- **Linear Algebra**: Matrix operations, eigenvalues/eigenvectors
- **Python**: Basic programming skills (variables, functions, plotting)

### Helpful (but not required)
- **Group Theory**: Symmetry operations and character tables
- **Solid State Physics**: Crystal structures and electronic bands
- **Atomic Physics**: Term symbols and electronic configurations

## 🎯 Who This Is For

### 🎓 **Students**
- Physics, chemistry, or materials science undergraduates/graduates
- Learning quantum mechanics applications to real systems
- Working on projects involving defects in solids

### 🔬 **Researchers**
- New to NV center physics
- Need comprehensive reference for energy level structure
- Developing quantum sensing experiments

### 👥 **Practitioners**
- Building NV-based sensors or quantum devices
- Need to understand underlying physics for optimization
- Teaching NV center physics to others

## 🌟 What Makes This Special

### 🎯 **Complete & Self-Contained**
Every concept is explained from scratch with clear physical reasoning. No external references required for basic understanding.

### 🔬 **Research-Grade Accuracy**
All parameters, energy levels, and transitions match current literature. Suitable for both education and research reference.

### 🎨 **Visual & Interactive**
Complex 3D structures and abstract quantum concepts are made concrete through interactive visualizations.

### 📚 **Progressive Learning**
Carefully designed learning path from basic concepts to advanced applications, with clear prerequisites for each section.

## 🚀 Quick Examples

### Visualize NV Center Structure
```python
# Run in nv_center_visualization.ipynb
create_nv_center_visualization(lattice_size=3, show_bonds=True)
```

### Calculate Zero-Field Splitting
```python
# From fundamentals notebook
D_gs = calculate_zero_field_splitting(
    spin_orbit=5.3e9,  # Hz
    dipolar=2.87e9     # Hz
)
print(f"Ground state ZFS: {D_gs/1e9:.2f} GHz")
```

### Simulate ODMR Spectrum
```python
# Advanced notebook example
spectrum = simulate_odmr(
    B_field=0.001,      # Tesla
    temperature=300,    # Kelvin
    microwave_power=1   # mW
)
plot_spectrum(spectrum)
```

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### 🐛 **Report Issues**
- Found an error in physics or code? Open an issue!
- Suggestions for better explanations or additional topics

### 📝 **Improve Content**
- Better explanations for difficult concepts
- Additional examples or applications
- Translations to other languages

### 💻 **Technical Improvements**
- Bug fixes in code
- Performance optimizations
- Better visualizations

### 📚 **Educational Enhancements**
- Additional exercises or problems
- Assessment quizzes
- Video explanations

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This educational resource builds upon decades of pioneering research by the NV center community. Special thanks to:

- **Neil Manson & John Harrison** (Australian National University) - Early spectroscopic studies
- **Jörg Wrachtrup** (University of Stuttgart) - Single NV detection and control
- **Mikhail Lukin** (Harvard University) - Quantum sensing applications
- **Ronald Walsworth** (University of Maryland) - Sensing protocol development

## 📧 Contact & Support

- **Issues**: Use GitHub issues for bug reports and feature requests
- **Questions**: Discussion tab for physics questions and learning support
- **Email**: [Your email for direct contact]

## 🌟 Star This Repository

If you find this resource helpful for learning or research, please give it a star! ⭐

It helps others discover this educational content and motivates continued development.

---

**Ready to explore the quantum world of NV centers?** 
👉 **[Start with the fundamentals](energy_diagram/nv_01_center_fundamentals.ipynb)** or **[jump right in online](https://mybinder.org/v2/gh/E-zClap/nv-center-notebook/main)**

