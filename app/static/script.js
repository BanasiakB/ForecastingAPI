// static/script.js

document.addEventListener('DOMContentLoaded', function() {
    // Initialize Select2 for variable selection
    $('#variable-select').select2();

    var dropArea = document.getElementById('drop-area');

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false)
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });

    function highlight() {
        dropArea.classList.add('highlight');
    }

    function unhighlight() {
        dropArea.classList.remove('highlight');
    }

    dropArea.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        var dt = e.dataTransfer;
        var files = dt.files;

        handleFiles(files);
    }

    function handleFiles(files) {
        var file = files[0];
        var formData = new FormData();
        formData.append('file', file);

        uploadFile(formData);
    }

    function uploadFile(formData) {
        fetch('/', {
            method: 'POST',
            body: formData
        })
        .then(response => response.text())
        .then(result => {
            document.getElementById('graph-container').innerHTML = result;
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }
});
