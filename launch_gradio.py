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
    color: #0a122a !important;
    font-family: 'Crimson Text', serif !important;
    font-size: 16px !important;
}

/* Fix loading screen and background */
.dark, .loading, .loading-wrap, #root {
    background-color: #ffffff !important;
    color: #0a122a !important;
}

/* Gradio app container */
.app, .gradio-app {
    background-color: #ffffff !important;
    color: #0a122a !important;
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
    color: #0a122a;
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

/* Gallery styling */
.gallery {
    background-color: #ffffff !important;
    padding: 1rem !important;
}

/* Sidebar experiment button styling */
.sidebar-experiment-btn {
    text-align: left !important;
    margin-bottom: 0.25rem !important;
    width: calc(100% - 1.5rem) !important;
}

/* Force override any dark mode classes */
.dark, .dark *:not(h1), [data-theme="dark"], [data-theme="dark"] *:not(h1) {
    background-color: #ffffff !important;
    color: #0a122a !important;
}

"""

# Create the Gradio interface at module level for hot swapping
demo = gr.Blocks(
    theme="monochrome", css=custom_css, title="HEP Foundation Results Viewer"
)

with demo:
    # Persistent expand button (positioned absolutely via CSS)
    expand_btn = gr.Button("▶", elem_classes=["expand-btn"], visible=False)

    with gr.Row():
        # Collapsible sidebar
        with gr.Column(scale=1, visible=True) as sidebar:
            with gr.Row():
                gr.Markdown("## Select Experiment")
                collapse_btn = gr.Button("◀", elem_id="sidebar-toggle", size="sm")

            # Get experiment folders
            experiment_folders = get_experiment_folders()
            experiment_buttons = []

            # State to track selected experiment
            selected_experiment_state = gr.State(value=None)

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
        with gr.Column(scale=4) as main_content:
            gr.Markdown("# HEP Foundation Results Viewer")

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

    # Prepare output components: state + all buttons + all sections
    output_components = [selected_experiment_state]

    # Add all experiment buttons to outputs
    for btn, _ in experiment_buttons:
        output_components.append(btn)

    # Add all section components
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
            fn=lambda current_selected, path=folder_path: select_experiment(
                path, current_selected
            ),
            inputs=[selected_experiment_state],
            outputs=output_components,
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
