// script.js
function uploadImages() {
    const fileInput = document.getElementById('file-input');
    const files = fileInput.files;

    if (!files.length) {
        return alert("Please choose files to upload.");
    }

    // 显示加载动画
    console.log("Showing loading animation");
    document.getElementById("loading-container").style.display = "block";
    document.getElementById("start-predict-btn").disabled = true; // 禁用按钮

    const formData = new FormData();
    Array.from(files).forEach(file => formData.append('files', file)); // Use 'files' as name

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        console.log("Response received");
        // 清空之前的结果
        const resultsContainer = document.getElementById('results');
        resultsContainer.innerHTML = ''; // Clear previous results

        // 隐藏加载动画
        console.log("Hiding loading animation");
        document.getElementById("loading-container").style.display = "none";
        document.getElementById("start-predict-btn").disabled = false; // 启用按钮

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
        // 隐藏加载动画
        console.log("Hiding loading animation (error)");
        document.getElementById("loading-container").style.display = "none";
        document.getElementById("start-predict-btn").disabled = false; // 启用按钮
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

// 动态更新指针位置
document.addEventListener('mousemove', function (e) {
    var cursor = document.getElementById('cursor');
    
    if (!cursor) {
        // 如果没有cursor元素，创建一个新的
        cursor = document.createElement('div');
        cursor.id = 'cursor';
        document.body.appendChild(cursor);
    }

    // 更新指针位置
    cursor.style.top = e.clientY + 'px';
    cursor.style.left = e.clientX + 'px';
});
