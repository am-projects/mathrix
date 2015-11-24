/*
document.getElementById('paper-submit').addEventListener('on-click', function() {
      document.forms[0].submit(); 
});
*/

function paperSubmit(element) {
  // document.forms[0].submit();
  element.getElementsByTagName('button')[0].click();
}