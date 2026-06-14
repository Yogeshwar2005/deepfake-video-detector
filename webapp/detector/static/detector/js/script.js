// Spinner
uploadForm = document.getElementById("upload-form");

if (uploadForm){
    uploadForm.addEventListener("submit", ()=>{
    document.getElementById("spinner").style.display="block";
    document.getElementById("detect-btn").disabled = true;
    document.getElementById("detect-btn").innerText="Detecting...";
});
}

// Theme Toggle
const themeToggle = document.getElementById("theme-toggle");

const savedTheme = localStorage.getItem("theme") || "light";
document.documentElement.setAttribute("data-bs-theme", savedTheme);

if (themeToggle){
    themeToggle.innerText = savedTheme === "dark"? "Light Mode" : "Dark Mode";
    themeToggle.addEventListener("click", ()=>{
        const currentTheme = document.documentElement.getAttribute("data-bs-theme");
        const newTheme = currentTheme === "dark"? "light": "dark";
        
        document.documentElement.setAttribute("data-bs-theme", newTheme);
        localStorage.setItem("theme", newTheme);
        themeToggle.innerText = newTheme ==="dark"? "Light Mode" : "Dark Mode";
    });


}

// Video Preview
const videoInput = document.querySelector("input[type='file']");
const videoPreview= document.getElementById("video-preview");

if(videoInput && videoPreview){
    videoInput.addEventListener("change", ()=>{
        const file = videoInput.files[0];

        if(file){
            const videoURL = URL.createObjectURL(file);
            videoPreview.src=videoURL;
            videoPreview.style.display = "block";
        }
        else{
            videoPreview.src = "";
            videoPreview.style.display="none";
        }
    });
}