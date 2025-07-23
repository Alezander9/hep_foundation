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
            folders["Test Results"] = test_folders

    # Check foundation experiments
    foundation_path = Path("_foundation_experiments")
    if foundation_path.exists():
        foundation_folders = [
            f
            for f in foundation_path.iterdir()
            if f.is_dir() and not f.name.startswith(".")
        ]
        if foundation_folders:
            folders["Foundation Experiments"] = foundation_folders

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
        "Regression Evaluation": [],
        "Signal Classification": [],
        "Anomaly Detection": [],
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

    return plots


# Custom CSS for clean light mode with orange accents
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Crimson+Text:ital,wght@0,400;0,600;1,400&display=swap');

/* Force light mode everywhere and remove all shadows/outlines */
* {
    box-shadow: none !important;
    outline: none !important;
}

/* Global light mode styling */
html, body {
    background-color: #ffffff !important;
    color: #2d3748 !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Remove Gradio's default max-width constraints for full width */
.gradio-container, .main, .wrap, .contain {
    max-width: none !important;
    width: 100% !important;
    background-color: #ffffff !important;
    color: #2d3748 !important;
    font-family: 'Crimson Text', serif !important;
    font-size: 16px !important;
}

/* Fix loading screen and background */
.dark, .loading, .loading-wrap, #root {
    background-color: #ffffff !important;
    color: #2d3748 !important;
}

/* Gradio app container */
.app, .gradio-app {
    background-color: #ffffff !important;
    color: #2d3748 !important;
    width: 100% !important;
    max-width: none !important;
}

/* Blocks and containers */
.block {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
}

/* Input fields and text areas */
input, textarea, select {
    background-color: #ffffff !important;
    color: #2d3748 !important;
    border: 1px solid #cbd5e0 !important;
    border-radius: 4px !important;
}

input:focus, textarea:focus, select:focus {
    border-color: #d36135 !important;
}

/* Buttons */
button {
    font-family: 'Crimson Text', serif !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
    transition: all 0.2s !important;
}

/* Primary buttons (orange) */
button.primary, .primary button {
    background-color: #d36135 !important;
    border-color: #d36135 !important;
    color: #ffffff !important;
}

button.primary:hover, .primary button:hover {
    background-color: #b8542d !important;
    border-color: #b8542d !important;
}

/* Secondary buttons */
button.secondary, .secondary button {
    background-color: #ffffff !important;
    border-color: #cbd5e0 !important;
    color: #4a5568 !important;
}

button.secondary:hover, .secondary button:hover {
    background-color: #f7fafc !important;
    border-color: #d36135 !important;
    color: #d36135 !important;
}

/* Headers and typography */
h1, h2, h3, h4, h5, h6 {
    color: #2d3748 !important;
    font-family: 'Crimson Text', serif !important;
    font-weight: 600 !important;
}

h1 {
    color: #d36135 !important;
    font-size: 2.5rem !important;
    text-align: center !important;
    margin-bottom: 2rem !important;
}

/* Markdown content */
.markdown {
    color: #2d3748 !important;
}
#markdown-no-border {
--block-border-width: 0 !important;
border: none !important;
}

/* Gallery styling */
.gallery {
    background-color: #f8f9fa !important;
    border-radius: 8px !important;
    padding: 1rem !important;
}

.gallery img {
    border-radius: 6px !important;
}

/* Sidebar styling */
.sidebar-section-title h4 {
    text-align: left !important;
    padding-left: 0.5rem !important;
    margin-bottom: 0.5rem !important;
    color: #4a5568 !important;
    font-weight: 600 !important;
}

.sidebar-experiment-btn {
    text-align: left !important;
    margin-left: 1.5rem !important;
    margin-bottom: 0.25rem !important;
    width: calc(100% - 1.5rem) !important;
}

/* Expand button styling */
.expand-btn {
    position: fixed !important;
    left: 10px !important;
    top: 120px !important;
    z-index: 1000 !important;
    border-radius: 50% !important;
    width: 40px !important;
    height: 40px !important;
    font-size: 16px !important;
    min-width: 40px !important;
    background-color: #d36135 !important;
    color: #ffffff !important;
    border: none !important;
}

.expand-btn:hover {
    background-color: #b8542d !important;
}

/* Section headers */
.section-header h2 {
    text-align: center !important;
    margin: 2rem 0 1rem 0 !important;
    border-bottom: 2px solid #d36135 !important;
    padding-bottom: 0.5rem !important;
    color: #d36135 !important;
}

/* Force override any dark mode classes */
.dark, .dark *, [data-theme="dark"], [data-theme="dark"] * {
    background-color: #ffffff !important;
    color: #2d3748 !important;
}

"""

# Create the Gradio interface at module level for hot swapping
demo = gr.Blocks(
    theme="monochrome", css=custom_css, title="HEP Foundation Results Viewer"
)

with demo:
    gr.Markdown(
        "# HEP Foundation Experiment Results Viewer",
        elem_classes=["markdown-no-border"],
    )

    # Persistent expand button (positioned absolutely via CSS)
    expand_btn = gr.Button("▶", elem_classes=["expand-btn"], visible=False)

    with gr.Row():
        # Collapsible sidebar
        with gr.Column(scale=1, visible=True) as sidebar:
            with gr.Row():
                gr.Markdown("### Experiment Selector")
                collapse_btn = gr.Button("◀", elem_id="sidebar-toggle", size="sm")

            # Get experiment folders
            experiment_folders = get_experiment_folders()
            experiment_buttons = []

            # Create buttons for each category
            for category, folders in experiment_folders.items():
                gr.Markdown(f"#### {category}", elem_classes=["sidebar-section-title"])
                for folder in folders:
                    folder_name = folder.name
                    folder_path = str(folder)
                    btn = gr.Button(
                        folder_name,
                        variant="secondary",
                        elem_classes=["sidebar-experiment-btn"],
                    )
                    experiment_buttons.append((btn, folder_path))

        # Main content area
        with gr.Column(scale=4) as main_content:
            gr.Markdown("### Results Display")
            selected_experiment = gr.Textbox(
                label="Selected Experiment",
                value="No experiment selected",
                interactive=False,
            )

            # Plot sections
            plot_sections = {}
            section_names = [
                "Dataset",
                "Backbone Model",
                "Regression Evaluation",
                "Signal Classification",
                "Anomaly Detection",
            ]

            for section_name in section_names:
                with gr.Row():
                    with gr.Column():
                        section_header = gr.Markdown(
                            f"## {section_name}",
                            visible=False,
                            elem_classes=["section-header"],
                        )
                        section_gallery = gr.Gallery(
                            label=f"{section_name} Plots",
                            show_label=False,
                            elem_id=f"gallery-{section_name.lower().replace(' ', '-')}",
                            columns=2,
                            rows=2,
                            height="auto",
                            visible=False,
                        )
                        plot_sections[section_name] = {
                            "header": section_header,
                            "gallery": section_gallery,
                        }

    # Toggle sidebar visibility
    def toggle_sidebar_collapse():
        return gr.update(visible=False), gr.update(visible=True)

    def toggle_sidebar_expand():
        return gr.update(visible=True), gr.update(visible=False)

    # Connect toggle buttons
    collapse_btn.click(toggle_sidebar_collapse, outputs=[sidebar, expand_btn])

    expand_btn.click(toggle_sidebar_expand, outputs=[sidebar, expand_btn])

    # Handle experiment button clicks
    def select_experiment(folder_path):
        plots_by_section = get_experiment_plots(folder_path)

        # Prepare outputs: experiment name + visibility and content for each section
        outputs = [folder_path]

        for section_name in section_names:
            section_plots = plots_by_section.get(section_name, [])
            if section_plots:
                # Show section with plots
                outputs.extend(
                    [
                        gr.update(visible=True),  # header
                        gr.update(value=section_plots, visible=True),  # gallery
                    ]
                )
            else:
                # Hide empty sections
                outputs.extend(
                    [
                        gr.update(visible=False),  # header
                        gr.update(value=[], visible=False),  # gallery
                    ]
                )

        return outputs

    # Prepare output components for all sections
    output_components = [selected_experiment]
    for section_name in section_names:
        output_components.extend(
            [
                plot_sections[section_name]["header"],
                plot_sections[section_name]["gallery"],
            ]
        )

    # Connect all experiment buttons
    for btn, folder_path in experiment_buttons:
        btn.click(
            lambda path=folder_path: select_experiment(path), outputs=output_components
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
