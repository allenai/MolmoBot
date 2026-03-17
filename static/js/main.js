function toggleAuthors() {
  const block = document.getElementById('authorBlock');
  const btn = document.querySelector('.authors-toggle');
  block.classList.toggle('open');
  btn.textContent = block.classList.contains('open') ? '▲ Hide Authors' : '▼ Show Authors';
}

// ── Video cell click ──────────────────────────────────────────────
const S3 = "https://ai2-prior-molmobot.s3.us-west-2.amazonaws.com/videos/droid_eval/";

const PROMPTS = {
  "workroom_v1/spoon_tray":       "put the wooden spoon on the light blue tray",
  "workroom_v1/spoon_box":        "put the wooden spoon in the wooden box",
  "workroom_v1/tape_tray":        "put the blue tape on the light blue tray",
  "workroom_v1/tape_box":         "put the blue tape in the wooden box",
  "workroom_v1/blue_mug_tray":    "put the blue mug on the light blue tray",
  "workroom_v1/blue_mug_box":     "put the blue mug in the wooden box",
  "workroom_v1/copper_mug_tray":  "put the copper mug on the light blue tray",
  "workroom_v1/copper_mug_box":   "put the copper mug in the wooden box",
  "workroom_v1/timer_tray":       "put the green timer on the light blue tray",
  "workroom_v1/timer_box":        "put the green timer in the wooden box",
  "kitchen_v3/apple_easy":        "put the apple in the brown bowl",
  "kitchen_v3/apple_hard":        "put the apple in the brown bowl",
  "kitchen_v3/mug_easy":          "put the mug in the brown bowl",
  "kitchen_v3/mug_hard":          "put the mug in the brown bowl",
  "kitchen_v3/banana_easy":       "put the banana in the brown bowl",
  "kitchen_v3/banana_hard":       "put the banana in the brown bowl",
  "kitchen_v3/mouse_easy":        "put the computer mouse in the brown bowl",
  "kitchen_v3/mouse_hard":        "put the computer mouse in the brown bowl",
  "kitchen_v3/clutter_brown":     "put the mug in the brown bowl",
  "kitchen_v3/clutter_black":     "put the mug in the black bowl",
  "bedroom_v1/pills_towel":       "put the pill bottle on the yellow towel",
  "bedroom_v1/pills_basket":      "put the pill bottle in the basket",
  "bedroom_v1/roller_towel":      "put the lint roller on the yellow towel",
  "bedroom_v1/roller_basket":     "put the lint roller in the basket",
  "bedroom_v1/banana_towel":      "put the banana on the yellow towel",
  "bedroom_v1/banana_basket":     "put the banana in the basket",
  "bedroom_v1/ball_towel":        "put the tennis ball on the yellow towel",
  "bedroom_v1/ball_basket":       "put the tennis ball in the basket",
  "bedroom_v1/clutter_towel":     "put the banana on the yellow towel",
  "bedroom_v1/clutter_basket":    "put the banana in the basket",
  "robomolmo_princeton/knife_board":           "put the knife on the cutting board",
  "robomolmo_princeton/banana_plate":          "move the toy banana on the plate",
  "robomolmo_princeton/marker_mug":            "put the marker into the mug",
  "robomolmo_princeton/scissors_bowl":         "pick up the scissor and place it inside the bowl",
  "robomolmo_princeton/carrot_basket":         "put the carrot into the basket",
  "robomolmo_princeton/knife_green_bowl":      "pick up the knife and place it inside the green bowl",
  "robomolmo_princeton/screwdriver_blue_bowl": "put the screwdriver in the blue bowl",
  "robomolmo_princeton/mouse_blue_bowl":       "place the mouse into the blue bowl",
  "robomolmo_princeton/mug_bowl":              "grasp the mug and put it inside the bowl",
  "robomolmo_princeton/marker_box":            "pick up the red marker and place it into the box",
};

// Outcome strings: 3 chars, 's'=success 'f'=failure, trials 1-2-3
const V = {
  "workroom_v1/spoon_tray/MolmoBot":"sss","workroom_v1/spoon_tray/MolmoBot-Img":"sss","workroom_v1/spoon_tray/MolmoBot-Pi0":"ffs","workroom_v1/spoon_tray/pi05-DROID":"ssf","workroom_v1/spoon_tray/pi0-DROID":"fff",
  "workroom_v1/spoon_box/MolmoBot":"fss","workroom_v1/spoon_box/MolmoBot-Img":"sss","workroom_v1/spoon_box/MolmoBot-Pi0":"fff","workroom_v1/spoon_box/pi05-DROID":"ssf","workroom_v1/spoon_box/pi0-DROID":"fff",
  "workroom_v1/tape_tray/MolmoBot":"sss","workroom_v1/tape_tray/MolmoBot-Img":"sss","workroom_v1/tape_tray/MolmoBot-Pi0":"sss","workroom_v1/tape_tray/pi05-DROID":"fss","workroom_v1/tape_tray/pi0-DROID":"ffs",
  "workroom_v1/tape_box/MolmoBot":"sss","workroom_v1/tape_box/MolmoBot-Img":"sss","workroom_v1/tape_box/MolmoBot-Pi0":"sss","workroom_v1/tape_box/pi05-DROID":"fff","workroom_v1/tape_box/pi0-DROID":"fff",
  "workroom_v1/blue_mug_tray/MolmoBot":"sss","workroom_v1/blue_mug_tray/MolmoBot-Img":"sss","workroom_v1/blue_mug_tray/MolmoBot-Pi0":"sss","workroom_v1/blue_mug_tray/pi05-DROID":"ffs","workroom_v1/blue_mug_tray/pi0-DROID":"fff",
  "workroom_v1/blue_mug_box/MolmoBot":"sss","workroom_v1/blue_mug_box/MolmoBot-Img":"ssf","workroom_v1/blue_mug_box/MolmoBot-Pi0":"sss","workroom_v1/blue_mug_box/pi05-DROID":"ffs","workroom_v1/blue_mug_box/pi0-DROID":"fff",
  "workroom_v1/copper_mug_tray/MolmoBot":"sss","workroom_v1/copper_mug_tray/MolmoBot-Img":"fss","workroom_v1/copper_mug_tray/MolmoBot-Pi0":"fss","workroom_v1/copper_mug_tray/pi05-DROID":"fff","workroom_v1/copper_mug_tray/pi0-DROID":"fff",
  "workroom_v1/copper_mug_box/MolmoBot":"fsf","workroom_v1/copper_mug_box/MolmoBot-Img":"fff","workroom_v1/copper_mug_box/MolmoBot-Pi0":"ffs","workroom_v1/copper_mug_box/pi05-DROID":"fff","workroom_v1/copper_mug_box/pi0-DROID":"fff",
  "workroom_v1/timer_tray/MolmoBot":"sss","workroom_v1/timer_tray/MolmoBot-Img":"sss","workroom_v1/timer_tray/MolmoBot-Pi0":"fff","workroom_v1/timer_tray/pi05-DROID":"fff","workroom_v1/timer_tray/pi0-DROID":"fff",
  "workroom_v1/timer_box/MolmoBot":"sss","workroom_v1/timer_box/MolmoBot-Img":"fsf","workroom_v1/timer_box/MolmoBot-Pi0":"sfs","workroom_v1/timer_box/pi05-DROID":"fff","workroom_v1/timer_box/pi0-DROID":"fff",

  "kitchen_v3/apple_easy/MolmoBot":"fsf","kitchen_v3/apple_easy/MolmoBot-Img":"sss","kitchen_v3/apple_easy/MolmoBot-Pi0":"ssf","kitchen_v3/apple_easy/pi05-DROID":"sss","kitchen_v3/apple_easy/pi0-DROID":"fff",
  "kitchen_v3/apple_hard/MolmoBot":"sfs","kitchen_v3/apple_hard/MolmoBot-Img":"sss","kitchen_v3/apple_hard/MolmoBot-Pi0":"sss","kitchen_v3/apple_hard/pi05-DROID":"fff","kitchen_v3/apple_hard/pi0-DROID":"fsf",
  "kitchen_v3/mug_easy/MolmoBot":"sss","kitchen_v3/mug_easy/MolmoBot-Img":"fss","kitchen_v3/mug_easy/MolmoBot-Pi0":"fss","kitchen_v3/mug_easy/pi05-DROID":"ssf","kitchen_v3/mug_easy/pi0-DROID":"fff",
  "kitchen_v3/mug_hard/MolmoBot":"sss","kitchen_v3/mug_hard/MolmoBot-Img":"sss","kitchen_v3/mug_hard/MolmoBot-Pi0":"fff","kitchen_v3/mug_hard/pi05-DROID":"sff","kitchen_v3/mug_hard/pi0-DROID":"fff",
  "kitchen_v3/banana_easy/MolmoBot":"sss","kitchen_v3/banana_easy/MolmoBot-Img":"sss","kitchen_v3/banana_easy/MolmoBot-Pi0":"sss","kitchen_v3/banana_easy/pi05-DROID":"sss","kitchen_v3/banana_easy/pi0-DROID":"sff",
  "kitchen_v3/banana_hard/MolmoBot":"sss","kitchen_v3/banana_hard/MolmoBot-Img":"sss","kitchen_v3/banana_hard/MolmoBot-Pi0":"sff","kitchen_v3/banana_hard/pi05-DROID":"sfs","kitchen_v3/banana_hard/pi0-DROID":"fff",
  "kitchen_v3/mouse_easy/MolmoBot":"sss","kitchen_v3/mouse_easy/MolmoBot-Img":"sss","kitchen_v3/mouse_easy/MolmoBot-Pi0":"sss","kitchen_v3/mouse_easy/pi05-DROID":"sss","kitchen_v3/mouse_easy/pi0-DROID":"sss",
  "kitchen_v3/mouse_hard/MolmoBot":"sss","kitchen_v3/mouse_hard/MolmoBot-Img":"sfs","kitchen_v3/mouse_hard/MolmoBot-Pi0":"ssf","kitchen_v3/mouse_hard/pi05-DROID":"fsf","kitchen_v3/mouse_hard/pi0-DROID":"fsf",
  "kitchen_v3/clutter_brown/MolmoBot":"fff","kitchen_v3/clutter_brown/MolmoBot-Img":"sss","kitchen_v3/clutter_brown/MolmoBot-Pi0":"fff","kitchen_v3/clutter_brown/pi05-DROID":"ssf","kitchen_v3/clutter_brown/pi0-DROID":"fff",
  "kitchen_v3/clutter_black/MolmoBot":"fff","kitchen_v3/clutter_black/MolmoBot-Img":"fsf","kitchen_v3/clutter_black/MolmoBot-Pi0":"fff","kitchen_v3/clutter_black/pi05-DROID":"ssf","kitchen_v3/clutter_black/pi0-DROID":"fff",

  "bedroom_v1/pills_towel/MolmoBot":"sss","bedroom_v1/pills_towel/MolmoBot-Img":"fff","bedroom_v1/pills_towel/MolmoBot-Pi0":"ssf","bedroom_v1/pills_towel/pi05-DROID":"fff","bedroom_v1/pills_towel/pi0-DROID":"fff",
  "bedroom_v1/pills_basket/MolmoBot":"sss","bedroom_v1/pills_basket/MolmoBot-Img":"fff","bedroom_v1/pills_basket/MolmoBot-Pi0":"fff","bedroom_v1/pills_basket/pi05-DROID":"fff","bedroom_v1/pills_basket/pi0-DROID":"fff",
  "bedroom_v1/roller_towel/MolmoBot":"sss","bedroom_v1/roller_towel/MolmoBot-Img":"ffs","bedroom_v1/roller_towel/MolmoBot-Pi0":"fss","bedroom_v1/roller_towel/pi05-DROID":"fff","bedroom_v1/roller_towel/pi0-DROID":"fff",
  "bedroom_v1/roller_basket/MolmoBot":"fff","bedroom_v1/roller_basket/MolmoBot-Img":"fss","bedroom_v1/roller_basket/MolmoBot-Pi0":"fff","bedroom_v1/roller_basket/pi05-DROID":"fff","bedroom_v1/roller_basket/pi0-DROID":"fff",
  "bedroom_v1/banana_towel/MolmoBot":"sss","bedroom_v1/banana_towel/MolmoBot-Img":"sss","bedroom_v1/banana_towel/MolmoBot-Pi0":"fff","bedroom_v1/banana_towel/pi05-DROID":"ffs","bedroom_v1/banana_towel/pi0-DROID":"fff",
  "bedroom_v1/banana_basket/MolmoBot":"fss","bedroom_v1/banana_basket/MolmoBot-Img":"sss","bedroom_v1/banana_basket/MolmoBot-Pi0":"fff","bedroom_v1/banana_basket/pi05-DROID":"fff","bedroom_v1/banana_basket/pi0-DROID":"fff",
  "bedroom_v1/ball_towel/MolmoBot":"sss","bedroom_v1/ball_towel/MolmoBot-Img":"sss","bedroom_v1/ball_towel/MolmoBot-Pi0":"ffs","bedroom_v1/ball_towel/pi05-DROID":"fff","bedroom_v1/ball_towel/pi0-DROID":"fff",
  "bedroom_v1/ball_basket/MolmoBot":"sss","bedroom_v1/ball_basket/MolmoBot-Img":"sfs","bedroom_v1/ball_basket/MolmoBot-Pi0":"fff","bedroom_v1/ball_basket/pi05-DROID":"fff","bedroom_v1/ball_basket/pi0-DROID":"fff",
  "bedroom_v1/clutter_towel/MolmoBot":"sss","bedroom_v1/clutter_towel/MolmoBot-Img":"sss","bedroom_v1/clutter_towel/MolmoBot-Pi0":"fff","bedroom_v1/clutter_towel/pi05-DROID":"ssf","bedroom_v1/clutter_towel/pi0-DROID":"fff",
  "bedroom_v1/clutter_basket/MolmoBot":"sss","bedroom_v1/clutter_basket/MolmoBot-Img":"sss","bedroom_v1/clutter_basket/MolmoBot-Pi0":"sfs","bedroom_v1/clutter_basket/pi05-DROID":"fff","bedroom_v1/clutter_basket/pi0-DROID":"fff",

  "robomolmo_princeton/knife_board/MolmoBot":"ssf","robomolmo_princeton/knife_board/MolmoBot-Img":"fss","robomolmo_princeton/knife_board/MolmoBot-Pi0":"sff","robomolmo_princeton/knife_board/pi05-DROID":"fss","robomolmo_princeton/knife_board/pi0-DROID":"fff",
  "robomolmo_princeton/banana_plate/MolmoBot":"sss","robomolmo_princeton/banana_plate/MolmoBot-Img":"sff","robomolmo_princeton/banana_plate/MolmoBot-Pi0":"sss","robomolmo_princeton/banana_plate/pi05-DROID":"sss","robomolmo_princeton/banana_plate/pi0-DROID":"fff",
  "robomolmo_princeton/marker_mug/MolmoBot":"ffs","robomolmo_princeton/marker_mug/MolmoBot-Img":"fff","robomolmo_princeton/marker_mug/MolmoBot-Pi0":"fff","robomolmo_princeton/marker_mug/pi05-DROID":"ffs","robomolmo_princeton/marker_mug/pi0-DROID":"fff",
  "robomolmo_princeton/scissors_bowl/MolmoBot":"ssf","robomolmo_princeton/scissors_bowl/MolmoBot-Img":"ffs","robomolmo_princeton/scissors_bowl/MolmoBot-Pi0":"fff","robomolmo_princeton/scissors_bowl/pi05-DROID":"ffs","robomolmo_princeton/scissors_bowl/pi0-DROID":"fff",
  "robomolmo_princeton/carrot_basket/MolmoBot":"fss","robomolmo_princeton/carrot_basket/MolmoBot-Img":"sss","robomolmo_princeton/carrot_basket/MolmoBot-Pi0":"fff","robomolmo_princeton/carrot_basket/pi05-DROID":"ffs","robomolmo_princeton/carrot_basket/pi0-DROID":"sff",
  "robomolmo_princeton/knife_green_bowl/MolmoBot":"fss","robomolmo_princeton/knife_green_bowl/MolmoBot-Img":"fsf","robomolmo_princeton/knife_green_bowl/MolmoBot-Pi0":"ffs","robomolmo_princeton/knife_green_bowl/pi05-DROID":"sff","robomolmo_princeton/knife_green_bowl/pi0-DROID":"fsf",
  "robomolmo_princeton/screwdriver_blue_bowl/MolmoBot":"sss","robomolmo_princeton/screwdriver_blue_bowl/MolmoBot-Img":"sss","robomolmo_princeton/screwdriver_blue_bowl/MolmoBot-Pi0":"sss","robomolmo_princeton/screwdriver_blue_bowl/pi05-DROID":"sfs","robomolmo_princeton/screwdriver_blue_bowl/pi0-DROID":"fsf",
  "robomolmo_princeton/mouse_blue_bowl/MolmoBot":"fss","robomolmo_princeton/mouse_blue_bowl/MolmoBot-Img":"ssf","robomolmo_princeton/mouse_blue_bowl/MolmoBot-Pi0":"fss","robomolmo_princeton/mouse_blue_bowl/pi05-DROID":"fsf","robomolmo_princeton/mouse_blue_bowl/pi0-DROID":"fsf",
  "robomolmo_princeton/mug_bowl/MolmoBot":"sss","robomolmo_princeton/mug_bowl/MolmoBot-Img":"sss","robomolmo_princeton/mug_bowl/MolmoBot-Pi0":"sss","robomolmo_princeton/mug_bowl/pi05-DROID":"sss","robomolmo_princeton/mug_bowl/pi0-DROID":"fff",
  "robomolmo_princeton/marker_box/MolmoBot":"sff","robomolmo_princeton/marker_box/MolmoBot-Img":"sfs","robomolmo_princeton/marker_box/MolmoBot-Pi0":"fss","robomolmo_princeton/marker_box/pi05-DROID":"ssf","robomolmo_princeton/marker_box/pi0-DROID":"fff",
};

const ENV_NAMES = {workroom_v1:"Workroom", kitchen_v3:"Kitchen", bedroom_v1:"Bedroom", robomolmo_princeton:"Office"};
const POLICY_NAMES = {"pi0-DROID":"π₀-DROID", "pi05-DROID":"π₀.₅-DROID"};

let selectedCell = null;

document.addEventListener("click", function(e) {
  const cell = e.target.closest("td.vc");
  if (!cell) return;

  // Deselect previous
  if (selectedCell && selectedCell !== cell) selectedCell.classList.remove("selected");

  // Toggle: clicking same cell again closes modal
  if (selectedCell === cell && document.getElementById("vidModal").classList.contains("open")) {
    closeModal();
    return;
  }

  selectedCell = cell;
  cell.classList.add("selected");

  const key = cell.dataset.vid;
  const parts = key.split("/");
  const env = parts[0], task = parts[1], policy = parts[2];
  const outcomes = (V[key] || "fff").split("");

  const envName = ENV_NAMES[env] || env;
  const taskName = task.replace(/_/g," ").replace(/\b\w/g, l => l.toUpperCase());
  const policyName = POLICY_NAMES[policy] || policy;

  document.getElementById("vidTitle").textContent = `${policyName}  ·  ${taskName}  ·  ${envName}`;

  const promptKey = `${env}/${task}`;
  const promptText = PROMPTS[promptKey] || "";
  let promptEl = document.getElementById("vidPrompt");
  if (!promptEl) {
    promptEl = document.createElement("p");
    promptEl.id = "vidPrompt";
    document.getElementById("vidContainer").parentNode.insertBefore(promptEl, document.getElementById("vidContainer"));
  }
  promptEl.innerHTML = promptText ? `<strong>Task:</strong> "${promptText}"` : "";
  promptEl.style.display = promptText ? "" : "none";

  const container = document.getElementById("vidContainer");
  container.innerHTML = "";
  outcomes.forEach((o, i) => {
    const isSucc = o === "s";
    const outcome = isSucc ? "succ" : "fail";
    const url = `${S3}${env}/${task}/${policy}_${i+1}_${outcome}.mp4`;
    const div = document.createElement("div");
    div.className = "vid-item " + (isSucc ? "vid-succ" : "vid-fail");
    div.innerHTML = `<video src="${url}" controls muted loop playsinline autoplay preload="auto"></video>
      <div class="vid-label">${isSucc ? "✓ Success" : "✗ Failure"}</div>`;
    container.appendChild(div);
  });

  document.getElementById("vidModal").classList.add("open");
});

function closeModal() {
  document.getElementById("vidModal").classList.remove("open");
  document.querySelectorAll("#vidContainer video").forEach(v => v.pause());
  if (selectedCell) { selectedCell.classList.remove("selected"); selectedCell = null; }
}

function closeModalOnBackdrop(e) {
  if (e.target === document.getElementById("vidModal")) closeModal();
}

document.addEventListener("keydown", e => { if (e.key === "Escape") closeModal(); });

// Wire up prompt tooltips on task header cells
document.querySelectorAll("th[data-task-key]").forEach(th => {
  const prompt = PROMPTS[th.dataset.taskKey];
  if (prompt) th.title = `Task: "${prompt}"`;
});
