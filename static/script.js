// static/script.js

// Function to generate a UUID
function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
        const r = Math.random() * 16 | 0,
            v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

document.getElementById('transcribeForm').onsubmit = async function (e) {
    e.preventDefault();

    // Hide previous results and show status
    document.getElementById('result').style.display = 'none';
    const statusDiv = document.getElementById('status');
    const statusMessage = document.getElementById('statusMessage');
    statusDiv.style.display = 'flex';
    statusMessage.innerText = 'Initializing...';

    // Generate a unique request ID
    const requestId = generateUUID();

    // Establish WebSocket connection
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(`${protocol}://${window.location.host}/ws/${requestId}`);

    ws.onopen = () => {
        console.log('WebSocket connection established.');
        statusMessage.innerText = 'Transcript will be downloaded automatically...';
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.status) {
            // Update status message
            statusMessage.innerText = data.status;
        }

        if (data.completed) {
            // Hide status and show results
            statusDiv.style.display = 'none';
            document.getElementById('transcriptText').innerText = data.transcription || 'No transcription available.';
            document.getElementById('summaryText').innerText = data.summary || 'No summary available.';

            // Handle diagram URL if available
            if (data.diagram_url) {
                const diagramImg = document.getElementById('diagram');
                if (diagramImg) {
                    diagramImg.src = data.diagram_url;
                    diagramImg.alt = 'Detected Diagram';
                    diagramImg.style.display = 'block';
                }
            } else {
                const diagramImg = document.getElementById('diagram');
                if (diagramImg) {
                    diagramImg.style.display = 'none';
                }
            }

            // Handle multiple images if available
            if (data.image_paths && data.image_paths.length > 0) {
                const imagesContainer = document.getElementById('imagesContainer');
                imagesContainer.innerHTML = ''; // Clear previous images
                data.image_paths.forEach((imgPath) => {
                    const img = document.createElement('img');
                    img.src = imgPath;
                    img.alt = 'Detected Image';
                    img.classList.add('detected-image'); // Add CSS class for styling
                    imagesContainer.appendChild(img);
                });
                imagesContainer.style.display = 'flex'; // Change to flex for better layout
            } else {
                document.getElementById('imagesContainer').style.display = 'none'; // Hide if no images
            }

            // Show the result section
            document.getElementById('result').style.display = 'block';
            // Show pop-up notification
            alert("Completed Successfully!");

            // Close WebSocket
            ws.close();
        }
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        statusMessage.innerText = 'An error occurred. Please try again.';
    };

    ws.onclose = () => {
        console.log('WebSocket connection closed.');
    };

    // Prepare form data with request ID
    const formData = new FormData(this);
    formData.append('request_id', requestId);

    // Send the POST request to initiate processing
    try {
        const response = await fetch('/transcribe', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json();
            statusMessage.innerText = `Error: ${errorData.detail}`;
            ws.close();
        }

        // Optionally handle response if needed
    } catch (error) {
        console.error('Error:', error);
        statusMessage.innerText = 'An error occurred while processing your request.';
        ws.close();
    }
};

// Modal Functionality
document.addEventListener('DOMContentLoaded', function () {
    const loginButton = document.getElementById('loginButton');
    const signupButton = document.getElementById('signupButton');
    const loginModal = document.getElementById('loginModal');
    const signupModal = document.getElementById('signupModal');
    const loginClose = document.getElementById('loginClose');
    const signupClose = document.getElementById('signupClose');

    // Open modals
    loginButton.onclick = () => loginModal.style.display = 'flex';
    signupButton.onclick = () => signupModal.style.display = 'flex';

    // Close modals
    loginClose.onclick = () => loginModal.style.display = 'none';
    signupClose.onclick = () => signupModal.style.display = 'none';

    // Close modal on outside click
    window.onclick = function (event) {
        if (event.target === loginModal) loginModal.style.display = 'none';
        if (event.target === signupModal) signupModal.style.display = 'none';
    };

    // Accessibility: Close modals with Escape key
    window.addEventListener('keydown', function (event) {
        if (event.key === 'Escape') {
            if (loginModal.style.display === 'flex') loginModal.style.display = 'none';
            if (signupModal.style.display === 'flex') signupModal.style.display = 'none';
        }
    });

    // Handle Login Form Submission
    const loginForm = document.getElementById('loginForm');
    loginForm.onsubmit = async function (e) {
        e.preventDefault();
        // Simulate a successful login for demonstration
        document.getElementById('loginSuccessMessage').style.display = 'block';
        setTimeout(() => {
            loginModal.style.display = 'none'; // Close the modal after showing message
            document.getElementById('loginSuccessMessage').style.display = 'none';
        }, 2000); // Message displayed for 2 seconds
    };

    // Handle Signup Form Submission
    const signupForm = document.getElementById('signupForm');
    signupForm.onsubmit = async function (e) {
        e.preventDefault();
        // Simulate a successful signup for demonstration
        document.getElementById('signupSuccessMessage').style.display = 'block';
        setTimeout(() => {
            signupModal.style.display = 'none'; // Close the modal after showing message
            document.getElementById('signupSuccessMessage').style.display = 'none';
        }, 2000); // Message displayed for 2 seconds
    };
});
