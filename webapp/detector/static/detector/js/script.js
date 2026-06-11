document.getElementById("upload-form").addEventListener("submit", ()=>{
    document.getElementById("spinner").style.display="block";
    document.getElementById("detect-btn").disabled = true;
    document.getElementById("detect-btn").innerText="Detecting..."
})