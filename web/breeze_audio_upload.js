import { app } from "../../../scripts/app.js";

// The core Comfy.AudioWidget extension injects its `audioUI` player widget
// only for a hardcoded list of core nodes (LoadAudio, SaveAudio, ...).
// Custom nodes with `audio_upload: true` get the AUDIOUPLOAD widget from
// Comfy.UploadAudio but never the audioUI widget it looks up at construction
// time, which makes node creation throw (updateUIWidget reads
// undefined.element) and the node silently fails to add. Mirror the core
// injection for our node, inserting audioUI immediately AFTER `audio` —
// construction order matters because AUDIOUPLOAD resolves it by name.
app.registerExtension({
  name: "BreezeTTS2.AudioUpload",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "BreezeTTS2Speaker") return;
    const required = nodeData?.input?.required;
    if (!required || required.audioUI) return;
    const rebuilt = {};
    for (const [key, value] of Object.entries(required)) {
      rebuilt[key] = value;
      if (key === "audio") rebuilt.audioUI = ["AUDIO_UI", {}];
    }
    if (!rebuilt.audioUI) rebuilt.audioUI = ["AUDIO_UI", {}];
    nodeData.input.required = rebuilt;
  },
});
