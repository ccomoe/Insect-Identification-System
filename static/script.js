// script.js
function uploadImages() {
    const fileInput = document.getElementById('file-input');
    const files = fileInput.files;

    if (!files.length) {
        return alert("Please choose files to upload.");
    }

    const formData = new FormData();
    Array.from(files).forEach(file => formData.append('files', file)); // Use 'files' as name

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        const resultsContainer = document.getElementById('results');
        resultsContainer.innerHTML = ''; // Clear previous results

        Object.entries(data).forEach(([filename, prediction]) => {
            const container = document.createElement('div');
            container.className = 'image-container';

            const img = document.createElement('img');
            img.src = `uploads/${filename}?${new Date().getTime()}`; // 加上当前时间戳作为查询参数
            img.alt = filename;

            const text = document.createElement('p');
            text.textContent = `Filename: ${filename}\nPredicted: ${prediction}`;

            container.appendChild(img);
            container.appendChild(text);
            resultsContainer.appendChild(container);
        });
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while uploading the files.');
    });
}

// 更新文件选择数量的函数
document.getElementById('file-input').addEventListener('change', function() {
    const fileCount = this.files.length;
    const fileCountText = document.getElementById('file-count');

    if (fileCount > 0) {
        fileCountText.textContent = `${fileCount} file(s) selected`;
        fileCountText.classList.remove('hidden'); // 显示文本
    } else {
        fileCountText.textContent = 'No files selected';
        fileCountText.classList.add('hidden'); // 隐藏文本
    }
});
