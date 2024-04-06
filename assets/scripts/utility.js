console.log('utility.js loaded');

window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clientside: {
        store_rect_coords: function () {
            const graphElement = document.getElementById('image');
            if (graphElement === null) {
                console.log('No element found with id "image"');
                return '';
            }
            rect = graphElement.getBoundingClientRect();
            //console.log(rect);
            return rect;
        }
    }
});