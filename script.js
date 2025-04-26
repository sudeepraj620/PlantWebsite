document.getElementById('imageUpload').addEventListener('change', function(event) {
    const file = event.target.files[0];
    const reader = new FileReader();
    reader.onload = function(e) {
      document.getElementById('preview').src = e.target.result;
    };
    reader.readAsDataURL(file);
  });
  
  async function sendImage() {
    const input = document.getElementById('imageUpload');
    if (!input.files[0]) return alert("Please upload an image.");
  
    const formData = new FormData();
    formData.append('file', input.files[0]);
  
    const response = await fetch('/predict', {
      method: 'POST',
      body: formData
    });
  
    const result = await response.json();
    document.getElementById('result').innerText = `Predicted Plant: ${result.prediction}`;
  }
  