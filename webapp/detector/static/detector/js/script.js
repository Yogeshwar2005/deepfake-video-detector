uploadForm = document.getElementById("upload-form");

if (uploadForm){
    uploadForm.addEventListener("submit", ()=>{
    document.getElementById("spinner").style.display="block";
    document.getElementById("detect-btn").disabled = true;
    document.getElementById("detect-btn").innerText="Detecting...";
});
}

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