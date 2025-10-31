// =========================================================================
// === VARIABLES GLOBALES Y SELECTORES DE ELEMENTOS ========================
// =========================================================================
const sidebar = document.getElementById('sidebar');
const camerasView = document.getElementById('cameras-view');
const gridContainer = document.getElementById('grid-container');
const singleView = document.getElementById('single-view');
const singleViewImage = document.getElementById('single-view-image');
const singleViewContainer = document.getElementById('single-view-container');

// Se declaran aquí, pero se asignan cuando el DOM está listo
let classSelectorPopup, classSelectionGrid;
let eventsView, syncView; // Se asignarán en DOMContentLoaded
let allEventsData = [];
let currentEditingCamera = null;

// =========================================================================
// === LÓGICA DE NAVEGACIÓN Y VISTAS ========================================
// =========================================================================
function toggleSidebar() {
  sidebar.classList.toggle('collapsed');
}

function showView(view) {
  // Ocultar todas las vistas principales
  camerasView.style.display = 'none';
  const configView = document.getElementById('config-view'); // Se obtiene aquí porque está en un include
  configView.style.display = 'none';
  eventsView.style.display = 'none';
  syncView.style.display = 'none';
  
  // Quitar la clase 'active' de todos los ítems de navegación
  document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));

  if (view === 'cameras') {
    camerasView.style.display = 'block';
    document.getElementById('nav-cameras').classList.add('active');
  } else if (view === 'config') {
    const configView = document.getElementById('config-view');
    configView.style.display = 'flex';
    document.getElementById('nav-config').classList.add('active');
    loadConfig();
  } else if (view === 'events') {
    eventsView.style.display = 'flex';
    document.getElementById('nav-events').classList.add('active');
    document.getElementById('event-date-select').valueAsDate = new Date();
    loadEventCameras();
    document.getElementById('event-gallery-container').innerHTML = '';
    document.getElementById('event-detail-view').classList.remove('visible');
  }
  else if (view === 'sync') {
    syncView.style.display = 'flex';
    document.getElementById('nav-sync').classList.add('active');
  }
}

function showSingleView(src, cameraName) {
  gridContainer.style.display = 'none';
  singleView.style.display = 'flex';
  singleViewImage.src = src;

  const originalControls = document.querySelector(`.video-wrapper[data-camera-name="${cameraName}"] .camera-controls`);
  const originalIcons = document.querySelector(`.video-wrapper[data-camera-name="${cameraName}"] .detected-classes-icons`);
  const originalMotionIcon = document.querySelector(`.video-wrapper[data-camera-name="${cameraName}"] .motion-icon`);

  if (originalControls) singleViewContainer.appendChild(originalControls.cloneNode(true));
  if (originalIcons) singleViewContainer.appendChild(originalIcons.cloneNode(true));
  if (originalMotionIcon) singleViewContainer.appendChild(originalMotionIcon.cloneNode(true));
}

function showGridView() {
  singleView.style.display = 'none';
  singleViewImage.src = "";
  singleViewContainer.querySelectorAll('.camera-controls, .detected-classes-icons, .motion-icon').forEach(el => el.remove());
  gridContainer.style.display = 'grid';
}

// =========================================================================
// === LÓGICA DE LA VISTA DE EVENTOS ========================================
// =========================================================================

async function loadEventCameras() {
  const select = document.getElementById('event-camera-select');
  select.innerHTML = '<option value="">-- Seleccionar Cámara --</option>';
  const res = await fetch('/api/event_cameras');
  const cameraIds = await res.json();
  cameraIds.forEach(id => {
    select.innerHTML += `<option value="${id}">Cámara #${id}</option>`;
  });
}

async function loadEvents() {
  const gallery = document.getElementById('event-gallery-container');
  const detailView = document.getElementById('event-detail-view');
  gallery.innerHTML = 'Cargando...';
  detailView.classList.remove('visible'); // Oculta el panel de detalles

  const camId = document.getElementById('event-camera-select').value;
  const date = document.getElementById('event-date-select').value;

  if (!camId) {
    gallery.innerHTML = 'Por favor, selecciona una cámara.';
    return;
  }

  const res = await fetch(`/api/events?camera_id=${camId}&date=${date}`);
  allEventsData = await res.json();

  if (allEventsData.length === 0) {
    gallery.innerHTML = 'No se encontraron eventos para esta selección.';
    return;
  }

  gallery.innerHTML = '';
  allEventsData.forEach((event, index) => {
    const thumb = document.createElement('div');
    thumb.className = 'event-thumbnail';
    thumb.innerHTML = `<img src="/static/${event.image_path}" alt="Evento">`;
    thumb.onclick = () => showEventDetail(index);
    gallery.appendChild(thumb);
  });
}

function showEventDetail(index) {
  const event = allEventsData[index];
  const detailView = document.getElementById('event-detail-view');
  const imgContainer = document.getElementById('event-detail-image-container');
  const jsonContainer = document.getElementById('event-detail-json');

  // Marcar la miniatura seleccionada
  document.querySelectorAll('.event-thumbnail.selected').forEach(el => el.classList.remove('selected'));
  const selectedThumb = document.querySelector(`#event-gallery-container .event-thumbnail:nth-child(${index + 1})`);
  if (selectedThumb) selectedThumb.classList.add('selected');

  imgContainer.innerHTML = `<img src="/static/${event.image_path}" alt="Detalle del evento">`;
  
  let jsonHtml = `<h4>Detalles del Evento</h4>`;
  jsonHtml += `<ul>`;
  jsonHtml += `<li><b>Hora:</b> ${new Date(event.timestamp_utc).toLocaleString()}</li>`;
  jsonHtml += `<li><b>Cámara:</b> ${event.camera_name} (ID: ${event.camera_id})</li>`;
  jsonHtml += `<li><b>Objetos Detectados:</b> ${Object.keys(event.objects_count).length}</li>`;
  jsonHtml += `</ul>`;

  event.objects.forEach((obj, i) => {
    jsonHtml += `<h4>Objeto #${i + 1}: ${obj.class_name}</h4>`;
    jsonHtml += `<ul>`;
    jsonHtml += `<li><b>Confianza:</b> ${(obj.confidence * 100).toFixed(1)}%</li>`;
    if (obj.appearance) { // Es una persona
      const upperColor = obj.appearance.upper_body.dominant_color_hex;
      const lowerColor = obj.appearance.lower_body.dominant_color_hex;
      jsonHtml += `<li><b>Parte Superior:</b> <span class="color-swatch" style="background-color:${upperColor}"></span> ${upperColor}</li>`;
      jsonHtml += `<li><b>Parte Inferior:</b> <span class="color-swatch" style="background-color:${lowerColor}"></span> ${lowerColor}</li>`;
    } else { // Otro objeto
      const color = obj.dominant_color_hex;
      jsonHtml += `<li><b>Color Dominante:</b> <span class="color-swatch" style="background-color:${color}"></span> ${color}</li>`;
    }
    jsonHtml += `</ul>`;
  });

  jsonContainer.innerHTML = jsonHtml;
  detailView.classList.add('visible');
}

// =========================================================================
// === LÓGICA DE LA VISTA DE SINCRONIZACIÓN ================================
// =========================================================================

function populateSyncForm() {
  document.getElementById('sync-enabled').checked = syncConfig.enabled;
  document.getElementById('sync-endpoint').value = syncConfig.endpoint || '';
  document.getElementById('sync-token').value = syncConfig.auth_token || '';
  document.getElementById('sync-mode').value = syncConfig.sync_mode || 'event';
  document.getElementById('sync-interval').value = syncConfig.interval_seconds || 30;

  // Mostrar u ocultar el campo de intervalo según el modo
  const mode = document.getElementById('sync-mode').value;
  document.getElementById('sync-interval-group').style.display = (mode === 'periodic' || mode === 'both') ? 'block' : 'none';
  document.getElementById('sync-mode').addEventListener('change', (e) => {
    document.getElementById('sync-interval-group').style.display = (e.target.value === 'periodic' || e.target.value === 'both') ? 'block' : 'none';
  });

  const cameraList = document.getElementById('sync-camera-list');
  cameraList.innerHTML = '';
  for (const camName in cameraConfigs) {
    const isChecked = syncConfig.cameras.includes(camName) || syncConfig.cameras.includes('all');
    cameraList.innerHTML += `
      <label>
        <input type="checkbox" value="${camName}" ${isChecked ? 'checked' : ''}>
        ${camName}
      </label>
    `;
  }
}

async function saveSyncConfig() {
  const selectedCameras = Array.from(document.querySelectorAll('#sync-camera-list input:checked')).map(cb => cb.value);

  const newConfig = {
    enabled: document.getElementById('sync-enabled').checked,
    endpoint: document.getElementById('sync-endpoint').value,
    auth_token: document.getElementById('sync-token').value,
    sync_mode: document.getElementById('sync-mode').value,
    interval_seconds: parseInt(document.getElementById('sync-interval').value, 10),
    cameras: selectedCameras.length === Object.keys(cameraConfigs).length ? ['all'] : selectedCameras
  };

  const res = await fetch('/api/sync_config', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(newConfig)
  });
  const result = await res.json();
  alert(result.message);
  // Actualizar la config global de JS
  Object.assign(syncConfig, newConfig);
}

// =========================================================================
// === LÓGICA DE LA VISTA DE CÁMARAS Y CONTROLES ===========================
// =========================================================================
async function loadConfig() {
  const res = await fetch('/api/config');
  document.getElementById('config-editor').value = await res.text();
}

async function saveConfig() {
  const content = document.getElementById('config-editor').value;
  const res = await fetch('/api/config', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ content })
  });
  const result = await res.json();
  alert(result.message);
}

async function toggleCamera(checkbox, cameraName) {
  const isEnabled = checkbox.checked;
  await fetch(`/api/camera/${cameraName}/toggle`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ enabled: isEnabled })
  });
  const videoWrapper = document.querySelector(`.video-wrapper[data-camera-name="${cameraName}"]`);
  videoWrapper.style.opacity = isEnabled ? 1 : 0.5;
}

async function toggleBbox(checkbox, cameraName) {
  const isVisible = checkbox.checked;
  cameraConfigs[cameraName].show_bbox = isVisible;
  await fetch(`/api/camera/${cameraName}/toggle_bbox`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ show: isVisible })
  });
}

async function toggleMotion(checkbox, cameraName) {
  const isVisible = checkbox.checked;
  cameraConfigs[cameraName].show_motion_contours = isVisible;
  await fetch(`/api/camera/${cameraName}/toggle_motion`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ show: isVisible })
  });
}

async function updateSensitivity(slider, cameraName) {
    const sensitivity = slider.value;
    cameraConfigs[cameraName].motion_sensitivity = sensitivity;
    await fetch(`/api/camera/${cameraName}/motion_sensitivity`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sensitivity: sensitivity })
    });
}

function openClassSelector(cameraName) {
  currentEditingCamera = cameraName;
  classSelectionGrid.innerHTML = '';
  const currentClasses = cameraConfigs[cameraName].detect_classes;

  for (const [id, name] of Object.entries(COCO_CLASSES)) {
    const isChecked = currentClasses.includes(parseInt(id));
    classSelectionGrid.innerHTML += `<label class="class-item"><input type="checkbox" value="${id}" ${isChecked ? 'checked' : ''}><span>${name}</span></label>`;
  }
  classSelectorPopup.style.display = 'flex';
}

function closeClassSelector() {
  classSelectorPopup.style.display = 'none';
  currentEditingCamera = null;
}

async function saveClassSelection() {
  if (!currentEditingCamera) return;
  const selectedClasses = Array.from(classSelectionGrid.querySelectorAll('input:checked')).map(input => input.value);
  await fetch(`/api/camera/${currentEditingCamera}/classes`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ classes: selectedClasses })
  });
  cameraConfigs[currentEditingCamera].detect_classes = selectedClasses.map(c => parseInt(c));
  updateClassIcons(currentEditingCamera);
  closeClassSelector();
}

function updateClassIcons(cameraName) {
    const container = document.getElementById(`icons-${cameraName}`);
    container.innerHTML = '';
    if (!cameraConfigs[cameraName] || !cameraConfigs[cameraName].detect_classes) return;
    cameraConfigs[cameraName].detect_classes.forEach(classId => {
        const className = COCO_CLASSES[classId] || 'unk';
        const icon = document.createElement('div');
        icon.className = 'class-icon';
        icon.id = `icon-${cameraName}-${classId}`;
        icon.textContent = className.substring(0, 4);
        icon.title = className;
        container.appendChild(icon);
    });
}

async function pollStatus() {
  try {
    const res = await fetch('/api/status');
    const statuses = await res.json();
    for (const cameraName in statuses) {
      if (!cameraConfigs[cameraName]) continue;
      const status = statuses[cameraName];
      const currentlyDetected = status.detected_classes;
      const configuredClasses = cameraConfigs[cameraName].detect_classes;
      configuredClasses.forEach(classId => {
        const icons = document.querySelectorAll(`#icon-${cameraName}-${classId}`);
        const isDetected = currentlyDetected.includes(classId);
        icons.forEach(icon => icon.classList.toggle('detected', isDetected));
      });
      const motionIcons = document.querySelectorAll(`#motion-icon-${cameraName}`);
      motionIcons.forEach(icon => icon.classList.toggle('detected', status.motion_detected));
    }
  } catch (e) {
    console.error("Error polling status:", e);
  }
}

// =========================================================================
// === INICIALIZACIÓN DE LA APLICACIÓN =====================================
// =========================================================================
// Inicializar estado al cargar la página
document.addEventListener('DOMContentLoaded', function() { // Asegurarse de que todo el DOM esté cargado
    eventsView = document.getElementById('events-view');
    syncView = document.getElementById('sync-view');
    classSelectorPopup = document.getElementById('class-selector-popup');
    classSelectionGrid = document.getElementById('class-selection-grid');

    // Configurar estado inicial de las cámaras
    document.querySelectorAll('.video-wrapper').forEach(wrapper => {
        const cameraName = wrapper.dataset.cameraName;
        wrapper.style.opacity = wrapper.querySelector('input[type="checkbox"]').checked ? 1 : 0.5;
        updateClassIcons(cameraName);
    });

    // Poblar el formulario de sincronización
    populateSyncForm();
    // Empezar a consultar el estado de las detecciones cada segundo
    setInterval(pollStatus, 1000);

    // Establecer la vista inicial
    showView('cameras');
});