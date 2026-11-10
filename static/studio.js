(() => {
const toggle=document.querySelector('.mobile-nav-toggle');
if(toggle)toggle.addEventListener('click',()=>{const open=document.querySelector('.sidebar').classList.toggle('is-visible');toggle.setAttribute('aria-expanded',String(open));});
const main=document.querySelector('.main');if(main){const footer=document.createElement('footer');footer.className='studio-footer';footer.innerHTML='<span>attend · Campus workspace</span><span>Studio edition</span>';main.appendChild(footer);}
})();
