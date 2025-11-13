// Sidebar UI logic 
let btn = document.querySelector('#btn');
let sidebar = document.querySelector('.sidebar');
let searchBtn = document.querySelector('.bx-search');
let listItem = document.querySelectorAll('.list-item');

btn.onclick = function() {
  sidebar.classList.toggle('active');
}
searchBtn.onclick = function() {
  sidebar.classList.toggle('active');
}
function activeLink() {
  listItem.forEach(item => item.classList.remove('active'));
  this.classList.add('active');
}
listItem.forEach(item => item.onclick = activeLink);

// Cartoonifier logic
const form = document.getElementById('cartoon-form');
const imageInput = document.getElementById('image-input');
const cartoonifyButton = document.getElementById('cartoonify-button');
const originalImage = document.getElementById('original-image');
const cartoonImage = document.getElementById('cartoon-image');
const loader = document.getElementById('loader');
const downloadBtn = document.getElementById('download-btn');

// Options
const styleSel = document.getElementById('style');
const edgeModeSel = document.getElementById('edge_mode');
const edgeThicken = document.getElementById('edge_thicken');
const edgeOpacity = document.getElementById('edge_opacity');
const kmeansK = document.getElementById('kmeans_k');

// Preview original when file chosen
imageInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  originalImage.src = url;
  originalImage.style.display = 'block';
  cartoonImage.style.display = 'none';
  downloadBtn.style.display = 'none';
});

// Submit form via fetch
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  if (!imageInput.files || !imageInput.files[0]) {
    alert('Please select an image first.');
    return;
  }

  loader.style.display = 'block';
  cartoonifyButton.disabled = true;

  try {
    const fd = new FormData();
    fd.append('image', imageInput.files[0]);
    fd.append('style', styleSel.value);
    fd.append('edge_mode', edgeModeSel.value);
    fd.append('edge_thicken', edgeThicken.value);
    fd.append('edge_opacity', edgeOpacity.value);
    fd.append('kmeans_k', kmeansK.value);

    const res = await fetch(form.action, { method: 'POST', body: fd });
    if (!res.ok) throw new Error('Failed to process image');
    const blob = await res.blob();
    const imgUrl = URL.createObjectURL(blob);
    cartoonImage.src = imgUrl;
    cartoonImage.style.display = 'block';
    downloadBtn.href = imgUrl;
    downloadBtn.style.display = 'inline-block';
  } catch (err) {
    console.error(err);
    alert(err.message || 'Something went wrong while cartoonifying.');
  } finally {
    loader.style.display = 'none';
    cartoonifyButton.disabled = false;
  }
});
