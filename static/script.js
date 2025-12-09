// script.js
let mainChart = null;
let errChart = null;

function resetOutput(){
  document.getElementById("output").classList.add("hidden");
  document.getElementById("r_val").textContent = "";
  document.getElementById("k_val").textContent = "";
  document.getElementById("rmse_val").textContent = "";
  document.querySelector("#val_table tbody").innerHTML = "";
  if(mainChart){ mainChart.destroy(); mainChart = null; }
  if(errChart){ errChart.destroy(); errChart = null; }
}

function runSim(){
  const country = document.getElementById("country").value;
  const start = parseInt(document.getElementById("start_year").value);
  const end = parseInt(document.getElementById("end_year").value);

  if(end <= start){ alert("Tahun akhir harus lebih besar dari tahun awal."); return; }
  const years = [];
  for(let y=start; y<=end; y++) years.push(y);
  //if(years.length < 2 || years.length > 3){ alert("Pilih rentang 2 sampai 3 tahun."); return; }

  document.getElementById("runBtn").disabled = true;
  fetch("/simulate", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({country: country, start_year: start, end_year: end})
  })
  .then(r => r.json())
  .then(res => {
    if(res.error){ alert("Error: " + res.error); document.getElementById("runBtn").disabled = false; return; }
    // show parameter
    document.getElementById("r_val").textContent = Number(res.r_fit).toFixed(6);
    document.getElementById("k_val").textContent = Number(res.K_fit).toLocaleString();
    document.getElementById("rmse_val").textContent = Number(res.rmse).toLocaleString();

    // fill validation table
    const tbody = document.querySelector("#val_table tbody");
    tbody.innerHTML = "";
    for(let i=0;i<res.obs_years.length;i++){
      const y = res.obs_years[i];
      const obs = res.obs_values[i];
      const pred = res.ys_at_obs[i];
      const err = res.errors_at_obs[i];
      const tr = `<tr>
                    <td>${y}</td>
                    <td>${Number(obs).toLocaleString()}</td>
                    <td>${Number(pred).toLocaleString()}</td>
                    <td>${Number(err).toLocaleString()}</td>
                  </tr>`;
      tbody.insertAdjacentHTML("beforeend", tr);
    }

    // draw main chart (Observasi points, Euler smooth curve, analytic)
    const ctx = document.getElementById("mainChart").getContext("2d");
    if(mainChart) mainChart.destroy();
    mainChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: res.t_smooth,
        datasets: [
          {
            label: 'Simulasi RK4',
            data: res.ys_smooth,
            borderColor: 'rgba(255,99,132,0.95)',
            borderWidth: 2,
            fill: false,
            pointRadius: 0
          },
          {
            label: 'Solusi Eksak (Analitik)',
            data: res.ys_exact,
            borderColor: 'rgba(124,92,255,0.95)',
            borderWidth: 2,
            borderDash: [6,3],
            fill:false,
            pointRadius: 0
          },
          {
            label: 'Observasi',
            data: res.obs_values.map((v,i)=> ({x: res.obs_years[i], y: res.obs_values[i]})),
            borderColor: 'rgba(0,0,0,0)',
            backgroundColor: 'rgba(0,0,0,1)',
            pointRadius: 5,
            showLine: false,
            type: 'scatter'
          }
        ]
      },
      options: {
        responsive: true,
        interaction: {mode: 'index'},
        scales: {
          x: { type: 'category', title:{display:true, text:'Tahun'} },
          y: { title:{display:true, text:'Populasi'} }
        },
        plugins: { legend: { position: 'bottom' } }
      }
    });

    // draw error chart
    const ctx2 = document.getElementById("errChart").getContext("2d");
    if(errChart) errChart.destroy();
    errChart = new Chart(ctx2, {
      type:'bar',
      data:{
        labels: res.obs_years,
        datasets:[{
          label:'Error Absolut',
          data: res.errors_at_obs,
          backgroundColor:'rgba(124,92,255,0.85)'
        }]
      },
      options:{
        responsive:true,
        scales:{ y:{ title:{display:true, text:'Error Absolut'} } },
        plugins:{ legend:{ display:false } }
      }
    });

    document.getElementById("output").classList.remove("hidden");
    document.getElementById("runBtn").disabled = false;

  })
  .catch(err=>{
    alert("Request failed: " + err);
    document.getElementById("runBtn").disabled = false;
  });
}
