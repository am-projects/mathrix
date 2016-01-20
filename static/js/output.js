document.addEventListener("change", showHide)

function showHide() {
  dom = document.getElementById('detail')
  dom.raised = dom.active ? 'raised' : '';
  document.getElementById('result-small').style.display =  dom.active ? 'none' : 'block';
  document.getElementById('result-full').style.display =  dom.active ? 'block' : 'none';
}


function hide(dom) {
  dom.style = (dom.style == "display: block") ? "display: none" : "display: block";
}
