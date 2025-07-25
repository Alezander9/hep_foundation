#!/usr/bin/env python3
"""
Gradio UI for viewing HEP Foundation experiment results.
"""

from pathlib import Path

import gradio as gr


def get_experiment_folders():
    """Get list of experiment folders from both test results and foundation experiments."""
    folders = {}

    # Check test results
    test_results_path = Path("_test_results/test_foundation_experiments")
    if test_results_path.exists():
        test_folders = [f for f in test_results_path.iterdir() if f.is_dir()]
        if test_folders:
            folders["Test Results"] = sorted(test_folders, key=lambda f: f.name)

    # Check foundation experiments
    foundation_path = Path("_foundation_experiments")
    if foundation_path.exists():
        foundation_folders = [
            f
            for f in foundation_path.iterdir()
            if f.is_dir() and not f.name.startswith(".")
        ]
        if foundation_folders:
            folders["Foundation Experiments"] = sorted(
                foundation_folders, key=lambda f: f.name
            )

    return folders


def get_experiment_plots(folder_path):
    """Scan experiment folder and organize plots by section."""
    if not folder_path or folder_path == "No experiment selected":
        return {}

    folder = Path(folder_path)
    if not folder.exists():
        return {}

    plots = {
        "Dataset": [],
        "Backbone Model": [],
        "Anomaly Detection": [],
        "Regression Evaluation": [],
        "Signal Classification": [],
    }

    # Dataset plots
    dataset_plots_dir = folder / "dataset_plots"
    if dataset_plots_dir.exists():
        for plot_file in dataset_plots_dir.glob("*.png"):
            plots["Dataset"].append(str(plot_file))

    # Training/Backbone model plots
    training_dir = folder / "training"
    if training_dir.exists():
        for plot_file in training_dir.glob("*.png"):
            plots["Backbone Model"].append(str(plot_file))

    # Testing plots
    testing_dir = folder / "testing"
    if testing_dir.exists():
        # Anomaly detection
        anomaly_dir = testing_dir / "anomaly_detection"
        if anomaly_dir.exists():
            # Check plots subdirectory
            anomaly_plots_dir = anomaly_dir / "plots"
            if anomaly_plots_dir.exists():
                for plot_file in anomaly_plots_dir.glob("*.png"):
                    plots["Anomaly Detection"].append(str(plot_file))
            else:
                # Check direct files
                for plot_file in anomaly_dir.glob("*.png"):
                    plots["Anomaly Detection"].append(str(plot_file))

        # Regression evaluation
        regression_dir = testing_dir / "regression_evaluation"
        if regression_dir.exists():
            for plot_file in regression_dir.glob("*.png"):
                plots["Regression Evaluation"].append(str(plot_file))

        # Signal classification
        signal_dir = testing_dir / "signal_classification"
        if signal_dir.exists():
            for plot_file in signal_dir.glob("*.png"):
                plots["Signal Classification"].append(str(plot_file))

    # Sort all plot lists alphabetically by filename
    for section in plots:
        plots[section].sort(key=lambda path: Path(path).name)

    return plots


# Custom CSS for clean light mode with orange accents
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Crimson+Text:ital,wght@0,400;0,600;1,400&display=swap');

/* Color Palette:
background (white): #ffffff
hover (blue-gray): #e2e1e0
text (dark): #0a122a
primary (orange): #d36135

 */

/* Force light mode everywhere and remove all shadows/outlines */
* {
    box-shadow: none !important;
    outline: none !important;
}

/* Global light mode styling */
html, body {
    background-color: #ffffff !important;
    color: #0a122a !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Remove Gradio's default max-width constraints for full width */
.gradio-container, .main, .wrap, .contain {
    max-width: none !important;
    width: 100% !important;
    background-color: #ffffff !important;
    font-family: 'Crimson Text', serif !important;
    font-size: 16px !important;
    min-width: 0 !important;
}

/* Ensure columns can flex properly */
.gr-column {
    min-width: 0 !important;
    flex-shrink: 1 !important;
}

/* Fix loading screen and background */
.dark, .loading, .loading-wrap, #root {
    background-color: #ffffff !important;
}

/* Gradio app container */
.app, .gradio-app {
    background-color: #ffffff !important;
    width: 100% !important;
    max-width: none !important;
}

/* Blocks and containers */
.block {
    background-color: #ffffff !important;
}

/* Unified button styling - white background, black content */
button {
    background-color: #ffffff !important;
    border-color: #ffffff !important;
    color: #0a122a !important;
    font-family: 'Crimson Text', serif !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
    transition: all 0.2s !important;
}

/* Button hover state - blue-gray background, keep black content */
button:hover {
    background-color: #e2e1e0 !important;
    border-color: #e2e1e0 !important;
    color: #0a122a !important;
}

/* Selected experiment button - orange content */
.selected-experiment {
    color: #d36135 !important;
}

/* Headers and typography */
h1, h2, h3, h4, h5, h6 {
    color: #0a122a !important;
    font-family: 'Crimson Text', serif !important;
    font-weight: 600 !important;
}

/* Special styling for H1 - centered and orange */
h1 {
    text-align: center !important;
    color: #d36135 !important;
}

/* Markdown content */
.markdown {
    color: #0a122a !important;
}

/* Image grid styling */
.image-grid {
    display: grid !important;
    grid-template-columns: repeat(2, 1fr) !important;
    gap: 1rem !important;
    padding: 1rem !important;
    background-color: #ffffff !important;
}

/* Sidebar styling */
.sidebar {
    border-right: 1px solid #e2e1e0 !important;
    padding-right: 0.5rem !important;
    min-width: 0 !important;
    flex-shrink: 1 !important;
}

/* Sidebar experiment button styling */
.sidebar-experiment-btn {
    text-align: left !important;
    margin-bottom: 0.1rem !important;
    min-width: 0 !important;
    width: 100% !important;
}

/* Force override any dark mode classes */
.dark, [data-theme="dark"] {
    background-color: #ffffff !important;
}

"""

# Create the Gradio interface at module level for hot swapping
demo = gr.Blocks(
    theme="monochrome", css=custom_css, title="HEP Foundation Results Viewer"
)

with demo:
    # State management
    selected_experiment_state = gr.State(value=None)
    sidebar_expanded_state = gr.State(value=True)

    with gr.Row():
        # Sidebar - conditionally sized
        with gr.Column(scale=4, elem_classes=["sidebar"]) as sidebar:
            # Sidebar toggle button
            sidebar_toggle_btn = gr.Button("↔", elem_classes=["sidebar-experiment-btn"])

            # Sidebar content (conditionally visible)
            with gr.Column(visible=True) as sidebar_content:
                gr.Markdown("## Select Experiment")

                # Get experiment folders
                experiment_folders = get_experiment_folders()
                experiment_buttons = []

                # Create buttons for each category
                for category, folders in experiment_folders.items():
                    gr.Markdown(f"### {category}")
                    for folder in folders:
                        folder_name = folder.name
                        folder_path = str(folder)
                        btn = gr.Button(
                            folder_name,
                            elem_classes=["sidebar-experiment-btn"],
                        )
                        experiment_buttons.append((btn, folder_path))

        # Main content area
        with gr.Column(scale=20) as main_content:
            gr.Markdown("# HEP Foundation Results Viewer")

            # Plot sections
            plot_sections = {}
            section_names = [
                "Dataset",
                "Anomaly Detection",
                "Backbone Model",
                "Regression Evaluation",
                "Signal Classification",
            ]

            for section_name in section_names:
                with gr.Row():
                    with gr.Column():
                        section_header = gr.Markdown(
                            f"## {section_name}",
                            visible=False,
                        )
                        with gr.Column(elem_classes=["image-grid"]):
                            # Create multiple image components for each section
                            section_images = []
                            for i in range(8):  # Max 8 images per section
                                img = gr.Image(
                                    label=f"{section_name} Plot {i + 1}",
                                    show_label=False,
                                    visible=False,
                                    container=False,
                                    height=None,  # Maintain aspect ratio
                                )
                                section_images.append(img)

                        plot_sections[section_name] = {
                            "header": section_header,
                            "images": section_images,
                        }

    # Handle experiment button clicks
    def select_experiment(folder_path, current_selected):
        plots_by_section = get_experiment_plots(folder_path)

        # Prepare outputs: new selected state + button updates + section visibility/content
        outputs = [folder_path]  # Update selected state

        # Update all experiment buttons - selected one gets orange class, others don't
        for btn, btn_folder_path in experiment_buttons:
            if btn_folder_path == folder_path:
                outputs.append(
                    gr.update(
                        elem_classes=["sidebar-experiment-btn", "selected-experiment"]
                    )
                )
            else:
                outputs.append(gr.update(elem_classes=["sidebar-experiment-btn"]))

        # Handle section visibility and content
        for section_name in section_names:
            section_plots = plots_by_section.get(section_name, [])
            if section_plots:
                # Show section header
                outputs.append(gr.update(visible=True))

                # Update individual images - show only images with plots
                section_images = plot_sections[section_name]["images"]

                for i, img in enumerate(section_images):
                    if i < len(section_plots):
                        # Show image with plot
                        outputs.append(gr.update(value=section_plots[i], visible=True))
                    else:
                        # Hide unused image slots
                        outputs.append(gr.update(visible=False))
            else:
                # Hide empty section header
                outputs.append(gr.update(visible=False))

                # Hide all images in this section
                section_images = plot_sections[section_name]["images"]
                for img in section_images:
                    outputs.append(gr.update(visible=False))

        return outputs

    # Prepare output components: state + all buttons + all sections
    output_components = [selected_experiment_state]

    # Add all experiment buttons to outputs
    for btn, _ in experiment_buttons:
        output_components.append(btn)

    # Add all section components
    for section_name in section_names:
        # Add section header
        output_components.append(plot_sections[section_name]["header"])

        # Add all individual images for this section
        output_components.extend(plot_sections[section_name]["images"])

    # Connect all experiment buttons
    for btn, folder_path in experiment_buttons:
        btn.click(
            fn=lambda current_selected, path=folder_path: select_experiment(
                path, current_selected
            ),
            inputs=[selected_experiment_state],
            outputs=output_components,
        )

    # Auto-select "001_Foundation_VAE_Model" from Test Results if it exists
    def auto_select_default():
        """Auto-select the default experiment on load."""
        target_name = "001_Foundation_VAE_Model"

        # Look for the target experiment in Test Results
        for btn, folder_path in experiment_buttons:
            folder = Path(folder_path)
            # Check if this is in test results and matches target name
            if (
                "test_foundation_experiments" in folder_path
                and folder.name == target_name
            ):
                # Trigger the selection
                return select_experiment(folder_path, None)

        # If not found, return empty updates
        return [None] + [gr.update() for _ in range(len(output_components) - 1)]

    # Set up auto-selection on load
    demo.load(
        fn=auto_select_default,
        inputs=[],
        outputs=output_components,
    )

    # Sidebar toggle functionality
    def toggle_sidebar(is_expanded):
        """Toggle sidebar between expanded and collapsed states."""
        new_expanded = not is_expanded

        if new_expanded:
            # Expanded state: scale=4, content visible
            return [
                new_expanded,  # Update state
                gr.update(scale=4),  # Sidebar column scale
                gr.update(visible=True),  # Sidebar content visibility
            ]
        else:
            # Collapsed state: scale=1, content hidden
            return [
                new_expanded,  # Update state
                gr.update(scale=1),  # Sidebar column scale
                gr.update(visible=False),  # Sidebar content visibility
            ]

    # Connect sidebar toggle button
    sidebar_toggle_btn.click(
        fn=toggle_sidebar,
        inputs=[sidebar_expanded_state],
        outputs=[sidebar_expanded_state, sidebar, sidebar_content],
    )


def main():
    """Main function to launch the Gradio interface."""
    # Launch the interface
    demo.launch(
        server_name="0.0.0.0",  # Allow access from other machines
        server_port=7860,  # Default Gradio port
        share=False,  # Don't create public link
        debug=True,  # Enable debug mode
    )


if __name__ == "__main__":
    main()
