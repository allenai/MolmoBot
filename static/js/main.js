// ── Carousel ──────────────────────────────────────────────────────
(function() {
  const track = document.querySelector('.carousel-track');
  if (!track) return;
  const slides = track.querySelectorAll('.carousel-slide');
  const dotsContainer = document.getElementById('carouselDots');
  let current = 0;

  slides.forEach((_, i) => {
    const dot = document.createElement('button');
    dot.className = 'carousel-dot' + (i === 0 ? ' active' : '');
    dot.setAttribute('aria-label', `Slide ${i + 1}`);
    dot.addEventListener('click', () => goTo(i));
    dotsContainer.appendChild(dot);
  });

  function goTo(idx) {
    slides[current].querySelector('video')?.pause();
    current = idx;
    track.style.transform = `translateX(-${current * 100}%)`;
    dotsContainer.querySelectorAll('.carousel-dot').forEach((d, i) =>
      d.classList.toggle('active', i === current)
    );
    document.getElementById('carouselPrev').disabled = current === 0;
    document.getElementById('carouselNext').disabled = current === slides.length - 1;
  }

  goTo(0);
  window.carouselMove = (dir) => goTo(Math.max(0, Math.min(slides.length - 1, current + dir)));
})();

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
const ENV_INLINE = {workroom_v1:"workroom", kitchen_v3:"kitchen", bedroom_v1:"bedroom", robomolmo_princeton:"office"};

let selectedCell = null;

function showVideos(cell) {
  if (selectedCell) selectedCell.classList.remove("selected");
  selectedCell = cell;
  cell.classList.add("selected");

  const key = cell.dataset.vid;
  const parts = key.split("/");
  const env = parts[0], task = parts[1], policy = parts[2];
  const outcomes = (V[key] || "fff").split("");
  const policyName = POLICY_NAMES[policy] || policy;
  const promptText = PROMPTS[`${env}/${task}`] || "";

  // Find the inline viewer for this table
  const inlineId = "vid-inline-" + ENV_INLINE[env];
  const inline = document.getElementById(inlineId);
  if (!inline) return;

  // Hide all other inline viewers and pause their videos
  document.querySelectorAll(".vid-inline").forEach(el => {
    if (el !== inline) {
      el.querySelectorAll("video").forEach(v => v.pause());
      el.classList.remove("active");
      el.innerHTML = "";
    }
  });

  const promptHtml = promptText
    ? `<span class="vid-inline-prompt"><strong>Task:</strong> "${promptText}"</span>`
    : "";

  let vidsHtml = outcomes.map((o, i) => {
    const isSucc = o === "s";
    const outcome = isSucc ? "succ" : "fail";
    const url = `${S3}${env}/${task}/${policy}_${i+1}_${outcome}.mp4`;
    return `<div class="vid-item ${isSucc ? "vid-succ" : "vid-fail"}">
      <video src="${url}" controls muted loop playsinline autoplay preload="auto"></video>
      <div class="vid-label">${isSucc ? "✓ Success" : "✗ Failure"}</div>
    </div>`;
  }).join("");

  inline.innerHTML = `
    <div class="vid-inline-header">
      <span class="vid-inline-title">${policyName}</span>
      ${promptHtml}
    </div>
    <div class="vid-grid">${vidsHtml}</div>
    <p class="vid-hint">All videos are fully autonomous and shown at 1× speed. Videos may take a moment to buffer.</p>
  `;
  inline.classList.add("active");
}

document.addEventListener("click", function(e) {
  const cell = e.target.closest("td.vc");
  if (!cell) return;
  showVideos(cell);
});

function copyBibtex() {
  const text = document.getElementById("bibtexBlock").textContent;
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.querySelector(".copy-btn");
    btn.innerHTML = '<i class="fa fa-check"></i> Copied!';
    setTimeout(() => { btn.innerHTML = '<i class="fa fa-copy"></i> Copy'; }, 2000);
  });
}

// ── Environment tabs ──────────────────────────────────────────────
document.querySelectorAll('.env-tab').forEach(tab => {
  tab.addEventListener('click', function() {
    const env = this.dataset.env;
    document.querySelectorAll('.env-tab').forEach(t => t.classList.toggle('active', t === this));
    document.querySelectorAll('.env-tab-panel').forEach(p => {
      const isActive = p.dataset.env === env;
      if (!isActive) {
        p.querySelectorAll('video').forEach(v => v.pause());
        p.querySelectorAll('.vid-inline').forEach(el => {
          el.classList.remove('active');
          el.innerHTML = '';
        });
        if (selectedCell && p.contains(selectedCell)) {
          selectedCell.classList.remove('selected');
          selectedCell = null;
        }
      }
      p.classList.toggle('active', isActive);
      if (isActive) {
        const firstCell = p.querySelector('td.vc');
        if (firstCell) showVideos(firstCell);
      }
    });
  });
});

// Wire up prompt tooltips on task header cells
document.querySelectorAll("th[data-task-key]").forEach(th => {
  const prompt = PROMPTS[th.dataset.taskKey];
  if (prompt) th.title = `Task: "${prompt}"`;
});

// Auto-select top-left cell of first table on load
const firstCell = document.querySelector("td.vc");
if (firstCell) showVideos(firstCell);

// ── DROID Results Interactive Bar Chart ───────────────────────────
(function() {
  const canvas = document.getElementById('droid-results-chart');
  if (!canvas) return;
  const tooltip = document.getElementById('droid-chart-tooltip');
  const ctx = canvas.getContext('2d');

  const data = [
    { label: 'MolmoBot',    value: 79.2, lo: 74.2, hi: 83.3, ours: true  },
    { label: 'MolmoBot-Img',value: 72.5, lo: 67.5, hi: 77.5, ours: true  },
    { label: 'MolmoBot-Pi0',value: 46.7, lo: 41.7, hi: 51.7, ours: true  },
    { label: 'π₀.₅-DROID',  value: 39.2, lo: 33.3, hi: 45.0, ours: false },
    { label: 'π₀-DROID',    value:  9.2, lo:  5.8, hi: 13.3, ours: false },
  ];

  const PINK      = '#e8529a';
  const PINK_DARK = '#c03a7a';
  const GRAY      = '#b0b0b0';
  const GRAY_DARK = '#888888';
  const MARGIN    = { top: 24, right: 24, bottom: 76, left: 52 };

  let cssW = 0, cssH = 320, dpr = 1;
  let barRects = [];
  let hoveredIdx = -1;

  function resize() {
    dpr = window.devicePixelRatio || 1;
    cssW = Math.min(canvas.parentElement.clientWidth, 640);
    cssH = 320;
    canvas.style.width  = cssW + 'px';
    canvas.style.height = cssH + 'px';
    canvas.width  = cssW * dpr;
    canvas.height = cssH * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function draw(hovIdx) {
    ctx.clearRect(0, 0, cssW, cssH);

    const chartW = cssW - MARGIN.left - MARGIN.right;
    const chartH = cssH - MARGIN.top  - MARGIN.bottom;
    const yScale = v => MARGIN.top + chartH * (1 - v / 100);

    const n      = data.length;
    const barW   = (chartW / n) * 0.58;
    const barStep = chartW / n;

    // Gridlines + y-axis labels
    ctx.setLineDash([]);
    for (let g = 0; g <= 100; g += 20) {
      const y = yScale(g);
      ctx.strokeStyle = '#e8e8e8';
      ctx.lineWidth   = 1;
      ctx.beginPath();
      ctx.moveTo(MARGIN.left, y);
      ctx.lineTo(MARGIN.left + chartW, y);
      ctx.stroke();

      ctx.fillStyle  = '#888';
      ctx.font       = '11px sans-serif';
      ctx.textAlign  = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(g + '%', MARGIN.left - 6, y);
    }

    // Axis lines
    ctx.strokeStyle = '#ccc';
    ctx.lineWidth   = 1;
    ctx.beginPath();
    ctx.moveTo(MARGIN.left, MARGIN.top);
    ctx.lineTo(MARGIN.left, yScale(0));
    ctx.lineTo(MARGIN.left + chartW, yScale(0));
    ctx.stroke();

    barRects = [];

    data.forEach((d, i) => {
      const cx   = MARGIN.left + barStep * i + barStep / 2;
      const x    = cx - barW / 2;
      const yTop = yScale(d.value);
      const h    = yScale(0) - yTop;
      const isHovered = i === hovIdx;

      barRects.push({ x, y: yTop, w: barW, h });

      // Bar
      ctx.fillStyle = d.ours
        ? (isHovered ? PINK_DARK : PINK)
        : (isHovered ? GRAY_DARK : GRAY);
      ctx.fillRect(x, yTop, barW, h);

      // Error bar
      const yLo  = yScale(d.lo);
      const yHi  = yScale(d.hi);
      const capW = barW * 0.28;
      ctx.strokeStyle = '#333';
      ctx.lineWidth   = isHovered ? 2 : 1.5;
      ctx.beginPath();
      ctx.moveTo(cx, yLo);
      ctx.lineTo(cx, yHi);
      ctx.stroke();
      ctx.beginPath(); ctx.moveTo(cx - capW/2, yLo); ctx.lineTo(cx + capW/2, yLo); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(cx - capW/2, yHi); ctx.lineTo(cx + capW/2, yHi); ctx.stroke();

      // Value label above the upper CI whisker
      ctx.fillStyle    = '#333';
      ctx.font         = isHovered ? 'bold 12px sans-serif' : '12px sans-serif';
      ctx.textAlign    = 'center';
      ctx.textBaseline = 'bottom';
      ctx.fillText(d.value + '%', cx, yHi - 4);

      // X-axis label (rotated)
      ctx.fillStyle    = isHovered ? '#111' : '#444';
      ctx.font         = isHovered ? 'bold 12px sans-serif' : '12px sans-serif';
      ctx.textAlign    = 'right';
      ctx.textBaseline = 'middle';
      ctx.save();
      ctx.translate(cx, yScale(0) + 10);
      ctx.rotate(-Math.PI / 6);
      ctx.fillText(d.label, 0, 0);
      ctx.restore();
    });
  }

  // Tooltip positioning relative to the wrapper div
  canvas.addEventListener('mousemove', function(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = cssW / rect.width;
    const mx = (e.clientX - rect.left) * scaleX;
    const my = (e.clientY - rect.top)  * scaleX;

    let found = -1;
    barRects.forEach(({ x, y, w, h }, i) => {
      if (mx >= x && mx <= x + w && my >= y && my <= y + h) found = i;
    });

    if (found !== hoveredIdx) {
      hoveredIdx = found;
      draw(hoveredIdx);
    }

    if (found >= 0) {
      const d = data[found];
      tooltip.textContent = `${d.value}%  (95% CI: ${d.lo}%–${d.hi}%)`;
      tooltip.style.display = 'block';
      // Position relative to canvas element, clamped to stay within wrapper
      const relX   = e.clientX - rect.left;
      const relY   = e.clientY - rect.top;
      const tipW   = tooltip.offsetWidth;
      const wrapW  = canvas.parentElement.clientWidth;
      let tipLeft  = relX + 14;
      if (tipLeft + tipW > wrapW) tipLeft = relX - tipW - 10;
      if (tipLeft < 0) tipLeft = 4;
      tooltip.style.left = tipLeft + 'px';
      tooltip.style.top  = (relY - 36) + 'px';
      canvas.style.cursor = 'default';
    } else {
      tooltip.style.display = 'none';
      canvas.style.cursor = '';
    }
  });

  canvas.addEventListener('mouseleave', function() {
    hoveredIdx = -1;
    tooltip.style.display = 'none';
    draw(-1);
  });

  resize();
  draw(-1);
  window.addEventListener('resize', function() { resize(); draw(hoveredIdx); });
})();
