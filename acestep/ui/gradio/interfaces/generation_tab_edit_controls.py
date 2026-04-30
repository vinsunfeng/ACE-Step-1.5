"""Flow-edit controls for the generation tab (#1156 PR-C).

Builds the Edit-mode-specific UI: target caption + target lyrics
textboxes, edit-type preset radio (``only_lyrics`` / ``remix`` /
``custom``), and the three edit-window sliders (``n_min`` / ``n_max``
/ ``n_avg``).  The whole group is wrapped in a ``gr.Group`` whose
visibility is toggled by the mode-change handler.
"""

from typing import Any

import gradio as gr


# Preset window values.  ``only_lyrics`` keeps the melody intact and
# changes lyrics; ``remix`` morphs the music with a tighter window
# earlier in the schedule.  ``custom`` exposes the sliders.
EDIT_TYPE_PRESETS = {
    "only_lyrics": {"n_min": 0.6, "n_max": 1.0},
    "remix":       {"n_min": 0.2, "n_max": 0.4},
}


def build_edit_controls() -> dict[str, Any]:
    """Create the Edit-mode control group used when ``mode == "Edit"``.

    Returns:
        Component map: ``edit_group`` (visibility toggle handle),
        ``edit_target_caption`` / ``edit_target_lyrics`` (target prompts),
        ``edit_type`` (preset radio), ``edit_n_min`` / ``edit_n_max`` /
        ``edit_n_avg`` (sliders).
    """

    with gr.Group(visible=False, elem_classes=["has-info-container"]) as edit_group:
        gr.HTML("<h5>Edit (flow-edit)</h5>")
        with gr.Row(equal_height=True):
            edit_target_caption = gr.Textbox(
                label="Target Caption",
                placeholder="Describe the desired output style/genre.",
                lines=4,
                max_lines=8,
                info="Empty = same as source caption (output will closely match source).",
            )
            edit_target_lyrics = gr.Textbox(
                label="Target Lyrics",
                placeholder="Lyrics for the edited output. Empty = keep source lyrics.",
                lines=4,
                max_lines=8,
            )
        edit_type = gr.Radio(
            choices=["only_lyrics", "remix", "custom"],
            value="only_lyrics",
            label="Edit Type",
            info=(
                "only_lyrics: keep melody, change lyrics (n_min=0.6, n_max=1.0). "
                "remix: change melody/genre (n_min=0.2, n_max=0.4). "
                "custom: enable the sliders below."
            ),
        )
        with gr.Row():
            edit_n_min = gr.Slider(
                label="edit_n_min",
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=0.6,
                interactive=False,  # Locked by preset; custom unlocks.
                info="Edit-window start (fraction of schedule).",
            )
            edit_n_max = gr.Slider(
                label="edit_n_max",
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=1.0,
                interactive=False,
                info="Edit-window end (fraction of schedule).",
            )
            edit_n_avg = gr.Slider(
                label="edit_n_avg",
                minimum=1,
                maximum=4,
                step=1,
                value=1,
                info="Monte-Carlo averages of V_delta per step. 1=fastest, 4=smoother.",
            )

    return {
        "edit_group": edit_group,
        "edit_target_caption": edit_target_caption,
        "edit_target_lyrics": edit_target_lyrics,
        "edit_type": edit_type,
        "edit_n_min": edit_n_min,
        "edit_n_max": edit_n_max,
        "edit_n_avg": edit_n_avg,
    }


def on_edit_type_change(edit_type_value: str):
    """Apply the preset window or enable sliders for ``custom``.

    Wired into ``edit_type.change(...)`` in the mode-wiring module.
    Returns updates for ``(edit_n_min, edit_n_max)`` — ``edit_n_avg`` is
    independent of the type preset.
    """
    if edit_type_value == "custom":
        return (
            gr.update(interactive=True),
            gr.update(interactive=True),
        )
    preset = EDIT_TYPE_PRESETS.get(edit_type_value, EDIT_TYPE_PRESETS["only_lyrics"])
    return (
        gr.update(value=preset["n_min"], interactive=False),
        gr.update(value=preset["n_max"], interactive=False),
    )
